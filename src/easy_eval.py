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
## NAIVE AVERAGING TOO
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    type=str, required=True, help="path to your config.yaml")
    p.add_argument("--ckpt",      type=str, required=True, help="path to state.pt")
    p.add_argument("--n_tasks",   type=int, default=5000, help="# of validation tasks")
    p.add_argument("--n_train",   type=int, default=40,   help="# of in-context points per task")
    p.add_argument("--lr",        type=float, default=0.05, help="LR for oracle MLE")
    p.add_argument("--max_steps", type=int, default=100000, help="max GD steps for oracle")
    p.add_argument("--tol",       type=float, default=1e-12, help="convergence tol for oracle")
    args = p.parse_args()

    # load config & build model
    # cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = build_model(cfg["model"]).to(device)

    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    model_conf = Namespace(**cfg["model"])
    model = build_model(model_conf).to(device)
    
    # load checkpoint
    state = torch.load(args.ckpt, map_location=device)
    
    
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print(f"Loaded model from {args.ckpt}")

    # sample a fresh validation set
    d = cfg["model"]["n_dims"]
    scale = cfg["training"]["task_kwargs"].get("scaling", 1.0)
    xs_all, ys_all, _ = sample_data(args.n_tasks, args.n_train+1, d, scale)
    xs_train, xs_test = xs_all[:, :-1], xs_all[:, -1:]
    ys_train, ys_test = ys_all[:, :-1], ys_all[:, -1:]

    # 1) In-context (transformer) evaluation
    with torch.no_grad():
        xs_t = xs_train.to(device)
        ys_t = ys_train.to(device)
        # model(...) returns a tensor of shape [batch, n_train]
        out = model(xs_t, ys_t)  
        # we only care about the prediction for the next point:
        last_loglam = out[:, -1]  # this is log λ̂
        loss_fn_model = PoissonNLLLoss(log_input=True, full=True, reduction="mean")
        # compare to ys_test; reshape last_loglam to [batch,1]
        model_loss = loss_fn_model(last_loglam.unsqueeze(-1), ys_test.to(device))
    print(f"Transformer model Poisson NLL (mean over tasks): {model_loss.item():.4f}")

    # 2) Oracle MLE baseline via pure gradient descent
    w_hat = torch.randn(args.n_tasks, d, 1, device=device, requires_grad=True)
    opt   = torch.optim.Adam([w_hat], lr=args.lr)
    loss_fn_base = PoissonNLLLoss(log_input=False, full=True, reduction="mean")

    prev = float("inf")
    for step in tqdm(range(1, args.max_steps+1), desc="Oracle GD"):
        logits = (xs_train.to(device) @ w_hat).squeeze(-1).clamp(max=3)
        pred   = torch.exp(logits)  # λ̂
        loss   = loss_fn_base(pred, ys_train.to(device))
        if abs(prev - loss.item()) < args.tol:
            print(f"Oracle converged at step {step} (Δloss={abs(prev-loss.item()):.2e})")
            break
        prev = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:
        print(f"Oracle did not fully converge in {args.max_steps} steps (final Δloss={abs(prev-loss.item()):.2e})")

    with torch.no_grad():
        logits_test = (xs_test.to(device) @ w_hat).squeeze(-1).clamp(max=3)
        pred_test   = torch.exp(logits_test)
        oracle_loss = loss_fn_base(pred_test, ys_test.to(device))
    print(f"Oracle baseline Poisson NLL (mean over tasks): {oracle_loss.item():.4f}")

if __name__ == "__main__":
    main()
