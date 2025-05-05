import os
from random import randint
import uuid
import copy
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
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
        r = args.training.task_kwargs.get("r", None),
    )
    print(f"[Init] Sampling new GLM each batch, function_type={glm_function}")

    pbar = tqdm(range(starting_step, args.training.train_steps))
    num_training_examples = args.training.num_training_examples
    
    for i in pbar:
        # 1) Sample data
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

        
        # print(f"[Step {i}] Function_type: {task.function_type}")

        ys = task.evaluate(xs)

        # print(f"[Step {i}] xs shape: {xs.shape}, ys shape: {ys.shape}")
        # print(f"[Step {i}] xs mean: {xs.mean().item():.4f}, std: {xs.std().item():.4f}")
        # print(f"[Step {i}] ys mean: {ys.mean().item():.4f}, std: {ys.std().item():.4f}")
        # print(f"[Step {i}] w_b mean: {task.w_b.mean():.4f}, std: {task.w_b.std():.4f}")


        # 3) Train on that fresh batch/task
        loss_func = task.get_training_metric()
        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)

        # print(f"[Step {i}] Output shape: {output.shape}, Loss: {loss:.4f}")

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

        # print(f"[Step {i}] Baseline loss: {baseline_loss:.4f}, Excess loss: {loss / baseline_loss:.4f}")

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(zip(point_wise_tags, point_wise_loss.cpu().numpy())),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "function_type": glm_function,
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
    GLM_TYPES=["linear", "sigmoid", "poisson", "logistic", "neg_binomial", "multinomial"]
    #optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    #loading from stopped training
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
    #dimension of x, batchsize
    n_dims = model.n_dims
    bsize = args.training.batch_size
    #Sampler from D_x (Like from Paper)
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        #Sampler from F (in our case a GLM)
        task_sampler = get_task_sampler(
            args.training.task,
            n_dims,
            bsize,
            function_type=GLM_TYPES[i % len(GLM_TYPES)],
            num_tasks=args.training.num_tasks,
        )
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "glm_type": GLM_TYPES[i % len(GLM_TYPES)]
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

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

    train(model, args, glm_function=args.training.task_kwargs.get("function_type"))

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
