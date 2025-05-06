import math
import torch
from torch.nn import functional as F
from torch.nn import PoissonNLLLoss

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "GLM": GLM,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims=n_dims, batch_size=batch_size, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class GLM(Task):
    def __init__(self, n_dims, batch_size, function_type="poisson", r=None, scale=None, seeds=None):
        super().__init__(n_dims, batch_size, None, seeds)
        self.function_type = function_type
        self.r = r
        self.scale = scale

        self.w_b = torch.randn(batch_size, n_dims, 1)
        if seeds is not None:
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(n_dims, 1, generator=generator)



    #Gets f(x) aka y label
    def evaluate(self, xs):
        B, K, D = xs.shape
        w_b = self.w_b.to(xs.device)
        z = (self.scale * (xs @ w_b)).squeeze(-1).clamp(-4, 4)

        if self.function_type == "linear":
            return z
        elif self.function_type == "sigmoid":
            return torch.sigmoid(z)
        elif self.function_type == "poisson":
            return torch.poisson(torch.exp(z))
        elif self.function_type == "logistic":
            return torch.bernoulli(torch.sigmoid(z))
        elif self.function_type == "neg_binomial":
            mu = torch.exp(z)
            r_tensor = torch.tensor(self.r, device=mu.device, dtype=mu.dtype)  # float r -> tensor
            logits = torch.log(r_tensor) - torch.log(mu)
            dist = torch.distributions.NegativeBinomial(
             total_count=r_tensor,
             logits=logits
         )
            return dist.sample()
        else:
            raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod   
    def generate_pool_dict(n_dims, num_tasks, function_type="poisson", **kwargs):
        return None  

    def get_training_metric(self):
        if self.function_type in ["linear", "sigmoid"]:
            return mean_squared_error
        elif self.function_type == "poisson":
            return PoissonNLLLoss(log_input=True, full=True)
        elif self.function_type  == "neg_binomial":
            r_val = self.r  # float
            def nb_nll_mean(preds, targets):
                mu = torch.exp(preds)
                r_tensor = torch.tensor(r_val, device=mu.device, dtype=mu.dtype)
                logits = torch.log(r_tensor) - torch.log(mu)
                dist = torch.distributions.NegativeBinomial(
                        total_count=r_tensor,
                        logits=logits
                    )
                return -dist.log_prob(targets).mean()
            return nb_nll_mean
        
        elif self.function_type == "logistic":
            return lambda input, target: F.binary_cross_entropy(torch.sigmoid(input), target)
        elif self.function_type == "multinomial":
            return lambda yhat, y: F.cross_entropy(yhat.view(-1, yhat.size(-1)), y.view(-1).long())
        else:
            raise NotImplementedError
    
