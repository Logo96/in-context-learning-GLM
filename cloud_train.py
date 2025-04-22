import copy
import torch
from models import build_model
from train import train, train_All_GLMS
from quinine import QuinineArgumentParser
from schema import schema
import os
import yaml

GLM_TYPES = ["linear", "sigmoid", "poisson", "logistic", "neg_binomial", "multinomial"]

def train_glm_individual(base_args, glm_type):
    print(f"Starting training for GLM type: {glm_type}")
    args = copy.deepcopy(base_args)
    args.training.task_kwargs["function_type"] = glm_type
    args.wandb.name = f"GLM_{glm_type}"
    args.out_dir = os.path.join(base_args.out_dir, f"{glm_type}_only")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file)

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args, glm_type)

def train_all_glms_together(base_args):
    print("Starting training GLMs jointly")
    args = copy.deepcopy(base_args)
    args.wandb.name = "GLM_MIXED"
    args.out_dir = os.path.join(base_args.out_dir, "all_glms")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file)

    model = build_model(args.model)
    model.cuda()
    model.train()
    
    train_All_GLMS(model, args)

def main():
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]

    glm_mode = os.environ.get("GLM_MODE", "joint")
    args.out_dir = os.path.join(args.out_dir, glm_mode)
    if glm_mode == "joint":
        train_all_glms_together(args)
    else:
        train_glm_individual(args, glm_type=glm_mode)

if __name__ == "__main__":
    main()
