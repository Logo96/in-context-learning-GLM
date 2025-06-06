# In-Context Learning of Generalized Linear Models (GLMs) with Transformers

This repository implements the experiments described in the project:

**Expanding the GLM Toolkit for Transformers as Statisticians**  

We explore the capability of transformer models to learn and distinguish between a wide range of **Generalized Linear Models (GLMs)** purely through **in-context learning**, without explicit gradient updates. Our approach extends prior work on linear and logistic regression by introducing **Poisson**, and **Negative Binomial** GLMs.
---

## Project Overview

Transformers have demonstrated the ability to approximate learning algorithms via in-context learning (ICL). This project investigates whether a single transformer model can:

- **Learn a variety of GLM types** from examples alone.
- **Select the correct GLM** family based on in-context patterns.
- **Adapt to overdispersion** in count data by implicitly choosing between Poisson and Negative Binomial models.

We show that a GPT-2-style transformer can learn to:
- Perform accurate regression across multiple GLM types.
- Select between families without any ground-truth family labels.
- Match or outperform naive baselines, and approach oracle-level GLM solutions in terms of likelihood.

---

## Code 

- **`train.py`** – Main training script. Handles curriculum scheduling, task sampling, optimizer setup, checkpointing, and W&B logging.
- **`tasks.py`** – Defines the `GLM` task class and sampling logic for different function families (`linear`, `logistic`, `poisson`, `neg_binomial`, `exponential`, etc.).
- **`glm_configs/`** – YAML configuration files specifying hyperparameters, curriculum, model architecture, and logging behavior.
- **`eval.ipnyb`** – Evaluation suite for all of our expirements
- **`fast_eval.py/ft_eval.py`** – A fast evaluation suite for evaluating several models at once. 

## Results
- **`plots/eval`** – Evaluation plots on validation set for poisson, negative binomial, and jointly trained models
- **`plots/train`** – Train plots for poisson, negative binomial, and jointly trained models

## Models

Huggingface: https://huggingface.co/icl-182

---

## Supported GLMs

| GLM Type           | Link Function        | Response Distribution | Data Type     |
|--------------------|----------------------|------------------------|---------------|
| Poisson Regression  | Exponential           | Poisson                | Count         |
| Negative Binomial   | Exponential           | NegBin (NB2)           | Count (overdispersed) |

---

## Methodology

For each training step:
1. A GLM function `f(x)` is sampled from a family (e.g. Poisson, Logistic).
2. A context of `k=40` in-context examples is generated from this function.
3. The transformer is trained to predict the label for a new input `x_{k+1}` for all `k<= 39` given previous context pairs
4. The loss is computed using the distribution-appropriate likelihood (e.g., PoissonNLLLoss).

All learning happens **without parameter updates at inference time**, relying solely on in-context adaptation.

---

## How to Train

Here's how to train the model:

1.  **Install dependencies**:
    ```bash
    source ./setup.sh
    ```
    NOTE: This setup is meant for linux systems only. 

2.  **Activate Conda Environment**:
    ```bash
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate icl
    ```

3.  **Run the training script**:
    Use the `train.py` script with your desired configuration file.
    ```bash
    python3 train.py --config glm_configs/poisson_0.32.yaml
    ```
    *(Replace `glm_configs/poisson_0.32.yaml` with the path to your specific config file.)*

4.  **Push to the Hugging Face Hub (Optional)**:
    Model results are saved to the `models/` directory. To push your trained model checkpoints to the Huggingface , you can use the push_to_hf.py file:

5.  **Run Evaluation**:
    Evaluate the trained model using the provided notebook in eval.ipynb. 
    You will typically need to specify the following parameters within the notebook (or pass them as variables):
    * `hf_model_id`: The ID of the model on Huggingface (e.g., `username/repo_name`).
    * `checkpoint_path`: The path to the model checkpoint file in the specified repo from above(e.g., `checkpoint-10000.pt`).
    * `eval_data_type`: The type of data to evaluate on (e.g., `poisson, negative binomial, etc. `).

    Other evaluation parameters like `scale` and `r` can usually be configured within the notebook itself if you need to deviate from the distribution from which the data was trained on. 

    Alternatively you can choose to use fast_eval.py or ft_eval.py with the required params, though this method is more for doing evals fast. fast_eval.py evaluates a model on the specific type of data it was trained on while ft_eval.py evaluates all provided models on all provided function types. Here are example commands for each.

    #### Example: `fast_eval.py`
    ```bash
        python fast_eval.py \
        --models nb-0.32-1.2 nb-0.32-2.5 nb-0.32-5.0 poisson-0.32 joint-0.32-1.2 joint-0.32-2.5 joint-0.32-5.0 \
        --titles "Negative Binomial (r=1.2)" "Negative Binomial (r=2.5)" "Negative Binomial (r=5.0)" "Poisson" "Joint (r=1.2)" "Joint (r=2.5)" "Joint (r=5.0)" \
        --out_dir ../plots/eval 
    ```

    #### Example: `ft_eval.py`
    ```bash
        python ft_eval.py \
        --models poisson-0.32 nb-0.32-1.2 nb-0.32-2.5 nb-0.32-5.0 \
        --model_titles "Poisson" "Negative Binomial (r=1.2)" "Negative Binomial (r=2.5)" "Negative
        Binomial (r=5.0)" \
        --function_types poisson neg_binomial neg_binomial neg_binomial \
        --rs 0.0 1.2 2.5 5.0 
    ```