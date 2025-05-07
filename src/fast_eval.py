#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import json
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tqdm import tqdm
from huggingface_hub import snapshot_download

from dotenv import load_dotenv
import os

load_dotenv()
# setting hf
hf_token = os.getenv("HF_TOKEN")
from huggingface_hub import login
login(hf_token)

# -----------------------------------------------------------------------------
# Data sampling & loss registry
# -----------------------------------------------------------------------------
LOSS_REGISTRY = {}
def register_loss(name):
    def deco(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return deco

@register_loss("poisson")
def poisson_loss_fn(pred, targets, **kw):
    return torch.nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")(pred, targets)

@register_loss("neg_binomial")
def nb_loss_fn(preds, targets, r, **kw):
    mu = torch.exp(preds)
    r_t = torch.tensor(r, device=mu.device, dtype=mu.dtype)
    logits = torch.log(r_t) - torch.log(mu)
    dist = torch.distributions.NegativeBinomial(total_count=r_t, logits=logits)
    return -dist.log_prob(targets)

def sample_data(n_tasks, n_points, d, ws=None, scale=1.0, loss_type="poisson", r=None, device="cpu"):
    xs = torch.randn(n_tasks, n_points, d, device=device)
    if ws is None:
        ws = scale * torch.randn(n_tasks, d, 1, device=device)
    logits = (xs @ ws).clamp(-4, 4)
    mu = torch.exp(logits)
    if loss_type == "poisson":
        ys = torch.poisson(mu).squeeze(-1)
    elif loss_type  == "neg_binomial":
        r_t = torch.tensor(r, device=device, dtype=mu.dtype)
        logits = torch.log(r_t) - torch.log(mu)
        dist = torch.distributions.NegativeBinomial(total_count=r_t, logits=logits)
        ys = dist.sample().squeeze(-1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return xs, ys, ws

# -----------------------------------------------------------------------------
# Evaluation routines
# -----------------------------------------------------------------------------
def get_logits(xs, ws):
    return (xs @ ws).squeeze(-1).clamp(-4, 4)

def evaluate_transformer(model, xs_all, ys_all, loss_type="poisson", r=None):
    loss_fn = LOSS_REGISTRY[loss_type]
    model.eval()
    with torch.no_grad():
        out = model(xs_all, ys_all)
        per_pos = loss_fn(out, ys_all, r=r).mean(dim=0)
    return per_pos.cpu().tolist()

def evaluate_oracle_gd(xs_all, ys_all, lr, max_steps, tol, scale=1.0, r=None, loss_type="poisson"):
    n_tasks, n_points, d = xs_all.shape
    all_losses = []
    loss_fn = LOSS_REGISTRY[loss_type]
    for t in tqdm(range(1, n_points), desc="GD Oracle"):
        xs_tr, ys_tr = xs_all[:, :t, :], ys_all[:, :t]
        xs_te, ys_te = xs_all[:, t:t+1, :], ys_all[:, t:t+1]
        w_hat = torch.randn(n_tasks, d, 1, device=xs_all.device, requires_grad=True)
        opt = torch.optim.Adam([w_hat], lr=lr)
        prev = float("inf")
        for _ in range(max_steps):
            logits = get_logits(xs_tr, w_hat)
            loss = loss_fn(logits, ys_tr, r=r).mean() + \
                   (0.5/scale**2) * w_hat.pow(2).sum()/(n_tasks*t)
            if abs(prev - loss.item()) < tol:
                break
            prev = loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            loss_te = loss_fn(get_logits(xs_te, w_hat), ys_te, r=r).mean().item()
        all_losses.append(loss_te)
    return all_losses

def evaluate_oracle_true(xs_all, ys_all, ws, loss_type="poisson", r=None):
    _, n_points, _ = xs_all.shape
    losses = []
    loss_fn = LOSS_REGISTRY[loss_type]
    for t in range(1, n_points):
        x_te = xs_all[:, t:t+1, :]
        y_te = ys_all[:, t:t+1]
        losses.append(loss_fn(get_logits(x_te, ws), y_te, r=r).mean().item())
    return losses

def evaluate_naive(ys_all, r=None):
    _, n_points = ys_all.shape
    losses = []
    for t in range(1, n_points):
        y_tr = ys_all[:, :t]
        y_te = ys_all[:, t:t+1]
        pred = y_tr.mean(dim=1, keepdim=True)
        eps = 1e-4
        losses.append(LOSS_REGISTRY["poisson"](torch.log(pred+eps), y_te, r=r).mean().item())
    return losses

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HF models, save loss plots + raw JSON"
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names in the repo to evaluate")
    parser.add_argument("--out_dir", default="outputs",
                        help="Root directory to save per-checkpoint folders")
    parser.add_argument("--n_tasks", type=int, default=10000,
                        help="Number of tasks for synthetic sampling")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate for GD oracle")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Max GD steps for the oracle")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="Convergence tolerance for GD oracle")
    args = parser.parse_args()

    torch.manual_seed(0)
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    for model_repo in args.models:
        # download & load config
        ckpt_dir = snapshot_download("icl-182/" + model_repo, repo_type="model")
        sys.path.append(ckpt_dir)
        cfg = yaml.safe_load(open(os.path.join(ckpt_dir, "config.yaml")))
        
        model_conf = SimpleNamespace(**cfg["model"])
        task_kwargs = cfg["training"]["task_kwargs"]
        scale = task_kwargs.get("scaling", 1.0)
        r_val = task_kwargs.get("r", None)
        n_context = cfg["model"]["n_positions"]

        func_types = task_kwargs.get("function_type",
                                    task_kwargs.get("loss_type", "poisson"))
        if isinstance(func_types, str):
            func_types = [func_types]

        base = os.path.splitext(model_repo)[0]
        folder = os.path.join(args.out_dir, base)
        os.makedirs(folder, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, "state.pt")

        # load model
        from models import build_model
        print(f"\nLoading {model_repo} …")
        state = torch.load(ckpt_path, map_location=device)
        model = build_model(model_conf).to(device)
        key = "model_state_dict" if "model_state_dict" in state else None
        model.load_state_dict(state.get(key, state))

        for ft in func_types:
            xs_all, ys_all, ws_true = sample_data(
                args.n_tasks, n_context, model_conf.n_dims,
                scale=scale, loss_type=ft, r=r_val, device=device
            )
            xs_all, ys_all = xs_all.to(device), ys_all.to(device)
            
            print(f"Evaluating {model_repo} with loss_type='{ft}' …")
            trans_losses = evaluate_transformer(model, xs_all, ys_all, ft, r_val)
            gd_losses    = evaluate_oracle_gd(xs_all, ys_all, args.lr,
                                              args.max_steps, args.tol,
                                              scale=scale, r=r_val, loss_type=ft)
            true_losses  = evaluate_oracle_true(xs_all, ys_all, ws_true, ft, r_val)
            naive_losses = evaluate_naive(ys_all.cpu(), r=r_val)

            # 1) save raw JSON
            json_out = {
                "transformer": trans_losses,
                "gd_oracle":  gd_losses,
                "true_oracle": true_losses,
                "naive":      naive_losses
            }
            json_path = os.path.join(folder, f"{base}_{ft}.json")
            with open(json_path, "w") as jf:
                json.dump(json_out, jf, indent=2)

            # 2) plot & save PNG
            ctx = list(range(1, n_context))
            plt.figure(figsize=(8,5))
            plt.plot(ctx, trans_losses[1:], label="Transformer", linewidth=2)
            plt.plot(ctx, gd_losses,    label="Baseline: Gradient Descent", linestyle="--", linewidth=2)
            plt.plot(ctx, true_losses,  label="Oracle: True Weights", linestyle=":", linewidth=2)

            loss_map = {"poisson": "Poisson", "neg_binomial": "Negative Binomial"}
            loss_name = loss_map.get(ft, ft)
            
            plt.xlabel("Context length", fontsize=12)
            plt.ylabel(f"{loss_name} Loss", fontsize=12)
            plt.title(f"{loss_name} Loss vs. Context Length", fontsize=14)
            plt.legend(loc="upper right", fontsize=12)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            # dynamic y-limits
            top_val = max(trans_losses)
            bot_val = min(true_losses)
            diff = top_val - bot_val
            plt.ylim(bot_val - diff*0.1, top_val + diff*0.36)

            plt.grid(True)
            plt.tight_layout()
            png_path = os.path.join(folder, f"{base}_{ft}.png")
            plt.savefig(png_path)
            plt.close()

            print(f" -> saved JSON to {json_path}")
            print(f" -> saved plot to {png_path}")

if __name__ == "__main__":
    main()
    