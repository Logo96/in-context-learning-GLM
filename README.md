# In-Context Learning of Generalized Linear Models (GLMs) with Transformers

This repository implements the experiments described in the project:

**Expanding the GLM Toolkit for Transformers as Statisticians**  

We explore the capability of transformer models to learn and distinguish between a wide range of **Generalized Linear Models (GLMs)** purely through **in-context learning**, without explicit gradient updates. Our approach extends prior work on linear and logistic regression by introducing **Poisson**, **Negative Binomial**, and **Multinomial** GLMs.
---

## Project Overview

Transformers have demonstrated the ability to approximate learning algorithms via in-context learning (ICL). This project investigates whether a single transformer model can:

- **Learn a variety of GLM types** from examples alone.
- **Select the correct GLM** family based on in-context patterns.
- **Adapt to overdispersion** in count data by implicitly choosing between Poisson and Negative Binomial models.

We show that a GPT-2-style transformer can learn to:
- Perform accurate regression/classification across multiple GLM types.
- Select between families without any ground-truth family labels.
- Match or outperform naive baselines, and approach oracle-level GLM solutions in terms of likelihood.

Huggingface: https://huggingface.co/icl-182

---

## Code 

- **`train.py`** – Main training script. Handles curriculum scheduling, task sampling, optimizer setup, checkpointing, and W&B logging.
- **`tasks.py`** – Defines the `GLM` task class and sampling logic for different function families (`linear`, `logistic`, `poisson`, `neg_binomial`, `exponential`, etc.).
- **`glm_configs/`** – YAML configuration files specifying hyperparameters, curriculum, model architecture, and logging behavior.
- **`scripts/`** – Optional scripts for running sweeps, evaluation, and training jobs on cloud infrastructure (GCP, Slurm, etc.).

---

## Supported GLMs

| GLM Type           | Link Function        | Response Distribution | Data Type     |
|--------------------|----------------------|------------------------|---------------|
| Linear Regression   | Identity              | Normal                 | Continuous    |
| Logistic Regression | Sigmoid               | Bernoulli              | Binary        |
| Poisson Regression  | Exponential           | Poisson                | Count         |
| Negative Binomial   | Exponential           | NegBin (NB2)           | Count (overdispersed) |
| Exponential         | Exponential           | Exponential            | Time-to-event |

---

## Methodology

For each training step:
1. A GLM function `f(x)` is sampled from a family (e.g. Poisson, Logistic).
2. A context of `k=40` in-context examples is generated from this function.
3. The transformer is trained to predict the label for a new input `x_{k+1}`.
4. The loss is computed using the distribution-appropriate likelihood (e.g., PoissonNLLLoss).

All learning happens **without parameter updates at inference time**, relying solely on in-context adaptation.

---

## How to Train

1. **Install dependencies**:
   ```bash
   source /setup.sh
2. ** Run train.py with appropriate config
