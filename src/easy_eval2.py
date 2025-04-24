import argparse
import torch
from torch.nn import PoissonNLLLoss
from tqdm import tqdm
import yaml
import os
from argparse import Namespace
from models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_data(n_tasks, n_points, d, scale=0.32):
    xs = torch.randn(n_tasks, n_points, d)
    ws = scale * torch.randn(n_tasks, d, 1)
    logits = xs @ ws
    ys = torch.poisson(torch.exp(logits.clamp(max=4))).squeeze(-1)
    return xs, ys, ws

def evaluate_transformer(model, ckpt_path, xs_train, ys_train, xs_test, ys_test):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()

    with torch.no_grad():
        out = model(xs_train, ys_train)
        last_loglam = out[:, -1]
        loss_fn = PoissonNLLLoss(log_input=True, full=True, reduction="mean")
        loss = loss_fn(last_loglam.unsqueeze(-1), ys_test)
    print(f"{os.path.basename(ckpt_path)} | Transformer Poisson NLL: {loss.item():.4f}")

def evaluate_oracle(xs_train, ys_train, xs_test, ys_test, lr, max_steps, tol):
    n_tasks, n_train, d = xs_train.shape
    w_hat = torch.randn(n_tasks, d, 1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w_hat], lr=lr)
    loss_fn = PoissonNLLLoss(log_input=False, full=True, reduction="mean")
    prev = float("inf")

    for step in tqdm(range(1, max_steps + 1), desc="Oracle GD"):
        logits = (xs_train @ w_hat).squeeze(-1).clamp(max=3)
        pred = torch.exp(logits)
        loss = loss_fn(pred, ys_train)
        if abs(prev - loss.item()) < tol:
            print(f"Oracle converged at step {step} (Δloss={abs(prev-loss.item()):.2e})")
            break
        prev = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_test = (xs_test @ w_hat).squeeze(-1).clamp(max=3)
        pred_test = torch.exp(logits_test)
        final_loss = loss_fn(pred_test, ys_test)
    print(f"Oracle baseline Poisson NLL: {final_loss.item():.4f}")

def evaluate_naive(ys_train, ys_test):
    naive_mean = ys_train.mean(dim=1, keepdim=True)
    loss_fn = PoissonNLLLoss(log_input=False, full=True, reduction="mean")
    loss = loss_fn(naive_mean, ys_test)
    print(f"Naive Mean Baseline Poisson NLL: {loss.item():.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir",  type=str, required=True)
    p.add_argument("--n_tasks",   type=int, default=5000)
    p.add_argument("--n_train",   type=int, default=40)
    p.add_argument("--lr",        type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--tol",       type=float, default=1e-12)
    args = p.parse_args()

    config_path = os.path.join(args.ckpt_dir, "config.yaml")
    cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
   
    model_conf = Namespace(**cfg["model"])
    model = build_model(model_conf).to(device)

    d = cfg["model"]["n_dims"]
    scale = cfg["training"]["task_kwargs"].get("scaling", 1.0)
    xs_all, ys_all, _ = sample_data(args.n_tasks, args.n_train + 1, d, scale)
    xs_train, xs_test = xs_all[:, :-1].to(device), xs_all[:, -1:].to(device)
    ys_train, ys_test = ys_all[:, :-1].to(device), ys_all[:, -1:].to(device)

    pt_files = sorted(f for f in os.listdir(args.ckpt_dir) if f.endswith(".pt"))
    for fname in pt_files:
        ckpt_path = os.path.join(args.ckpt_dir, fname)
        evaluate_transformer(model, ckpt_path, xs_train, ys_train, xs_test, ys_test)

    evaluate_oracle(xs_train, ys_train, xs_test, ys_test, args.lr, args.max_steps, args.tol)
    evaluate_naive(ys_train, ys_test)

if __name__ == "__main__":
    main()
