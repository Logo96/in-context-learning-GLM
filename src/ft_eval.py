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

from fast_eval import (
    sample_data,
    evaluate_oracle_gd,
    evaluate_oracle_true,
    evaluate_naive,
    evaluate_transformer,
)

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare HF models + baselines in one plot + JSON per function type"
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model repo names (in icl-182)")
    parser.add_argument("--model_titles", nargs="+", required=True,
                        help="Titles for each model (order matches --models)")
    parser.add_argument("--out_dir", default="outputs",
                        help="Root directory to save JSON and PNG per function type")
    parser.add_argument("--rs",   nargs="+", type=float, required=True,
                    help="Dispersion params for neg_binomial (match --function_types; ignored for poisson)")
    parser.add_argument("--function_types", nargs="+", required=True,
                        choices=["poisson", "neg_binomial"],
                        help="List of loss/function types (e.g. poisson neg_binomial)")
    
    parser.add_argument("--scale", default=0.32,  type=float,
                        help="Scaling factors for weight sampling (must match --function_types)")
    parser.add_argument("--n_tasks", type=int, default=10000,
                        help="Number of tasks for synthetic sampling")
    parser.add_argument("--lr",      type=float, default=0.05,
                        help="Learning rate for GD oracle")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Max GD steps for GD oracle")
    parser.add_argument("--tol",     type=float, default=1e-10,
                        help="Convergence tol for GD oracle")
    args = parser.parse_args()


    torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load first model config to get dims & context
    first_ckpt = snapshot_download("icl-182/" + args.models[0], repo_type="model")
    cfg = yaml.safe_load(open(os.path.join(first_ckpt, "config.yaml")))
    n_context = cfg["model"]["n_positions"]
    n_dims    = cfg["model"]["n_dims"]

    scale = args.scale
    # iterate over each function type / scale / r
    if len(args.rs) != len(args.function_types):
        raise ValueError("Number of rs must match number of function types")
    
    for ft, r in zip(args.function_types, args.rs):
        print(f"\n=== Processing function type: {ft} (scale={scale}, r={r}) ===")

        # sample data for this function type
        xs_all, ys_all, ws_true = sample_data(
            args.n_tasks, n_context, n_dims,
              scale=scale, loss_type=ft, r=r, device=device
        )
      
        xs_all, ys_all = xs_all.to(device), ys_all.to(device)

        # compute baselines
        print("Computing baselines…")
        gd_losses    = evaluate_oracle_gd(xs_all, ys_all,
                                          args.lr, args.max_steps, args.tol,
                                          scale, r, ft)
        true_losses  = evaluate_oracle_true(xs_all, ys_all,
                                            ws_true, ft, r)

        # set up plot + results dict
        plt.figure(figsize=(8,5), constrained_layout=True)
        ctx = list(range(1, n_context))
        plt.plot(ctx, gd_losses,   label="GD Oracle",    linestyle="--", linewidth=2)
        plt.plot(ctx, true_losses, label="True Oracle",  linestyle=":",  linewidth=2)
        results = {
            "GD Oracle":   gd_losses,
            "True Oracle": true_losses,
        }

        # evaluate and plot each model
        for repo, title in zip(args.models, args.model_titles):
            print(f"  • Loading and evaluating {repo}…")
            ckpt_dir = snapshot_download("icl-182/" + repo, repo_type="model")
            sys.path.append(ckpt_dir)
            state = torch.load(os.path.join(ckpt_dir, "state.pt"),
                                map_location=device)
            from models import build_model
            mc = SimpleNamespace(**yaml.safe_load(
                    open(os.path.join(ckpt_dir, "config.yaml"))
                )["model"])
            model = build_model(mc).to(device)
            key = "model_state_dict" if "model_state_dict" in state else None
            model.load_state_dict(state.get(key, state))

            losses = evaluate_transformer(model, xs_all, ys_all, ft, r)
            results[title] = losses
            plt.plot(ctx, losses[1:], label=title, linewidth=2)

        # auto y–limits
        all_vals = [v for arr in results.values() for v in arr]
        top_val, bot_val = max(all_vals), min(all_vals)
        diff = top_val - bot_val
        plt.ylim(bot_val - diff * 0.1, top_val + diff * 1.04)

        # labels & legend
        loss_map = {"poisson": "Poisson", "neg_binomial": "Negative Binomial"}
        loss_name = loss_map.get(ft, ft)
        plt.xlabel("Context length", fontsize=12)
        plt.ylabel(f"{loss_name} loss", fontsize=12)
        
        title = f"Validation Loss for {loss_name} Data"
        if loss_name == "Negative Binomial":
            title += f" (r={r})"
            
        plt.title(title, fontsize=14)
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()

        # save into subfolder
        ft_dir = os.path.join(args.out_dir, ft)
        os.makedirs(ft_dir, exist_ok=True)
        
        save = str(ft)
        if save == "neg_binomial":
            save += f"_{r}"
        
        json_path = os.path.join(ft_dir, f"{save}.json")
        png_path  = os.path.join(ft_dir, f"{save}.png")

        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=2)
        plt.savefig(png_path)
        plt.close()

        print(f"Saved JSON → {json_path}")
        print(f"Saved plot  → {png_path}")

if __name__ == "__main__":
    main()
