import os
import random
from random import randint
import uuid
import copy
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import mu_from_logits
from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, xs, ys, optimizer, loss_func):
    xs = xs.to(device)
    ys = ys.to(device)
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def train(model, args, glm_function=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate,
                              betas=(0.9, 0.95))
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for _ in range(state["train_step"] + 1):
            curriculum.update()
        print(f"[Resume] Resumed from step {starting_step}")

    n_dims = model.n_dims
    bsize  = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)

    task_sampler = get_task_sampler(
        "GLM",
        n_dims,
        bsize,
        function_type=glm_function,
        num_tasks=args.training.num_tasks,
        scale = args.training.task_kwargs.get("scaling", 1),
    )
    print(f"[Init] Sampling new GLM each batch, function_type={glm_function}")

    pbar = tqdm(range(starting_step, args.training.train_steps))
    num_training_examples = args.training.num_training_examples
    
    for i in pbar:
        data_sampler_args = {}
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler()
        ys = task.evaluate(xs)
        loss_func = task.get_training_metric()
        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)

        if i % 500 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"[Grad {i}] norm {grad_norm:.3f}")

        point_wise_tags     = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss     = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        with torch.no_grad():
            mu_hat = mu_from_logits(output)
            mu_hat_mean = mu_hat.mean().item()
            ys_mean = ys.to(mu_hat).mean().item()
            r_mean = task.r.mean().item()
            r_std = task.r.std().item()
            per_ep_var = ys.var(dim=1)
            per_ep_mean = ys.mean(dim=1)
            overdispersion = per_ep_var / (per_ep_mean + 1e-6)
            overdispersion_mean = overdispersion.mean().item()

        print(f"[Step {i}] μ̂ mean: {mu_hat_mean:.4f}, y mean: {ys_mean:.4f}, r mean: {r_mean:.2f}, overdispersion: {overdispersion_mean:.2f}")

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(zip(point_wise_tags, point_wise_loss.cpu().numpy())),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "function_type": glm_function,
                    "mu_hat_mean": mu_hat_mean,
                    "y_mean": ys_mean,
                    "r_mean": r_mean,
                    "overdispersion": overdispersion_mean,
                    "grad_norm": grad_norm if i % 500 == 0 else None,
                },
                step=i,
            )

        curriculum.update()
        pbar.set_description(f"loss {loss:.4f}")

        if i % args.training.save_every_steps == 0 and not args.test_run:
            torch.save(
                {
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_step":           i,
                },
                state_path,
            )
            print(f"[Checkpoint] Saved model at step {i}")

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, f"model_{i}.pt"),
            )
            print(f"[Snapshot] Kept model checkpoint at step {i}")


def train_All_GLMS(model, args):
    GLM_TYPES = args.training["task_kwargs"]["function_type"]
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)
    curriculum  = Curriculum(args.training.curriculum)
    state_path  = os.path.join(args.out_dir, "state.pt")
    starting_step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for _ in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize  = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    pbar   = tqdm(range(starting_step, args.training.train_steps))
    num_training_examples = args.training.num_training_examples

    for i in pbar:
        family = random.choice(GLM_TYPES)
        task_sampler = get_task_sampler(
            args.training.task,
            n_dims,
            bsize,
            function_type=family,
            num_tasks=args.training.num_tasks,
        )

        data_sampler_args, task_sampler_args = {}, {}
        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"]  = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        ).to(device)

        task   = task_sampler(**task_sampler_args)
        ys     = task.evaluate(xs).to(device)
        output = model(xs, ys)

        loss_func = task.get_training_metric()
        loss      = loss_func(output, ys)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)

        baseline_loss = (
            sum(max(curriculum.n_dims_truncated - ii, 0) for ii in range(curriculum.n_points))
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            fam_loss_dict = {
                f"family_loss/{ft}": (loss.item() if ft == family else None)
                for ft in GLM_TYPES
            }
            wandb.log(
                {
                    "overall_loss": loss.item(),
                    "excess_loss":  (loss / baseline_loss).item(),
                    "pointwise/loss": dict(zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())),
                    "n_points": curriculum.n_points,
                    "n_dims":   curriculum.n_dims_truncated,
                    "glm_type": family,
                    **fam_loss_dict,
                },
                step=i,
            )
            print(f"[{i}] family: {family}")

        curriculum.update()
        pbar.set_description(f"{family} | loss {loss.item():.4f}")

        if i % args.training.save_every_steps == 0 and not args.test_run:
            torch.save(
                {
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_step":           i,
                },
                state_path,
            )

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))




def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(name=args.wandb.name)
        # wandb.init(
        #     dir=args.out_dir,
        #     project=args.wandb.project,
        #     entity=args.wandb.entity,
        #     config=args.__dict__,
        #     notes=args.wandb.notes,
            #   name=args.wandb.name,
        #     resume=True,
        # )

    model = build_model(args.model).to(device)
    model.train()

    #train(model, args, glm_function=args.training.task_kwargs.get("function_type"))
    train_All_GLMS(model, args)
    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
