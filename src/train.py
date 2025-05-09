import os
from random import randint
import uuid
import copy
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from huggingface_hub import hf_hub_download

import wandb

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")

    # load from pretrained 
    if args.model.hf_pretrain_path:
        hf_state_path = hf_hub_download(
            repo_id=args.model.hf_pretrain_path, 
            filename="state.json",
            repo_type="model"
        )
        state = torch.load(hf_state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        # don't update starting step or curriculum 
        print(f"Start from pretrained model: {args.model.hf_pretrain_path}")

    elif os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for _ in range(state["train_step"] + 1):
            curriculum.update()
        print(f"[Resume] Resumed from step {starting_step}")

    n_dims      = model.n_dims
    batch_size  = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    
    r = args.training.task_kwargs.get("r", None)
    scale = args.training.task_kwargs.get("scaling", None)
    
    func_types = args.training.task_kwargs.get("function_type", None)
    # backwards compatible with single model training
    func_types = func_types if isinstance(func_types, list) else [func_types]
    
    loss_funcs = []
    for ft in func_types:
        dummy = get_task_sampler(
            "GLM", n_dims, batch_size,
            function_type=ft,
            num_tasks=args.training.num_tasks,
            scale=scale, r=r,
        )()
        loss_funcs.append(dummy.get_training_metric())
        
    pbar = tqdm(range(starting_step, args.training.train_steps))

    for step in pbar:
        xs = data_sampler.sample_xs(
            curriculum.n_points, batch_size, curriculum.n_dims_truncated
        ).to(device)

        ft_ids = torch.randint(0, len(func_types),
                               (batch_size,), device=device)

        ys = torch.empty(batch_size, curriculum.n_points, device=device)
        with torch.no_grad():
            for ft_id, ft in enumerate(func_types):
                mask = ft_ids == ft_id
                if not mask.any():
                    continue

                m = mask.sum().item()           
                sampler = get_task_sampler(
                    "GLM", n_dims, m,
                    function_type=ft,
                    num_tasks=args.training.num_tasks,
                    scale=scale, r=r,
                )()                  

                ys[mask] = sampler.evaluate(xs[mask])

        # -------- forward & loss --------
        optimizer.zero_grad()
        preds = model(xs, ys)          

        per_family_loss = {}
        total_loss = 0.0
        for ft_id, ft in enumerate(func_types):
            mask = ft_ids == ft_id
            if not mask.any():
                continue
            loss_val = loss_funcs[ft_id](preds[mask], ys[mask])
            per_family_loss[ft] = loss_val.detach().item()
            total_loss += loss_val

        total_loss /= len(func_types)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step  % 100 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"[Grad {step}] norm {grad_norm:.3f}")

        # -------- W&B logging --------
        if step % args.wandb.log_every_steps == 0 and not args.test_run:
            log_dict = {
                "joint_loss": total_loss.item(),
                "n_points"  : curriculum.n_points,
                "n_dims"    : curriculum.n_dims_truncated,
                "grad_norm": grad_norm if step % 100 == 0 else None,
            }
            # add individual GLM losses
            for ft, val in per_family_loss.items():
                log_dict[f"{ft}_loss"] = val
            wandb.log(log_dict, step=step)

        curriculum.update()
        pbar.set_description(
            " | ".join([f"{ft}:{per_family_loss[ft]:.3f}"
                         for ft in func_types if ft in per_family_loss]) +
            f" || joint:{total_loss:.3f}"
        )

        if step % args.training.save_every_steps == 0 and not args.test_run:
            torch.save(
                {
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_step":           step,
                },
                state_path,
            )
            print(f"[Checkpoint] Saved model at step {step}")

        if (
            args.training.keep_every_steps > 0
            and step % args.training.keep_every_steps == 0
            and not args.test_run
            and step > 0
        ):
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, f"model_{step}.pt"),
            )
            print(f"[Snapshot] Kept model checkpoint at step {step}")


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(name=args.wandb.name)
      

    model = build_model(args.model).to(device)
    model.train()

    train(model, args)

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