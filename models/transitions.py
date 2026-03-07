import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import to_torch_const, extract


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))
# %%


class DiscreteTransition(nn.Module):
    def __init__(self, num_timesteps, num_classes, noise_schedule=None, s=None, betas=None, prior_probs=None):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        if noise_schedule is not None:
            # atom type diffusion schedule in log space
            if noise_schedule == 'cosine':
                alphas_v = cosine_beta_schedule(self.num_timesteps, s)
                print('cosine v alpha schedule applied!')
            else:
                raise NotImplementedError
            log_alphas_v = np.log(alphas_v)
            log_alphas_cumprod_v = np.cumsum(log_alphas_v)
            self.log_alphas_v = to_torch_const(log_alphas_v)
            self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
            self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
            self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))
            if prior_probs is None:
                uniform_probs = -np.log(num_classes).repeat(num_classes)[None, :]  # (1, num_classes)
                self.prior_probs = to_torch_const(uniform_probs)
            else:
                print('prior types are used!')
                log_probs = np.log(prior_probs.clip(min=1e-30))
                self.prior_probs = to_torch_const(log_probs)
        else:
            assert betas is not None
            alphas_v = 1. - betas
            log_alphas_v = np.log(alphas_v)
            log_alphas_cumprod_v = np.cumsum(log_alphas_v)
            self.log_alphas_v = to_torch_const(log_alphas_v)
            self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
            self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
            self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

            if prior_probs is None:
                self.prior_probs = to_torch_const(np.log(np.ones(num_classes) / num_classes)[None, :])
            elif prior_probs == 'absorb':  # absorb all states into the first one
                init_prob = 0.01 * np.ones(num_classes)
                init_prob[0] = 1
                self.prior_probs = to_torch_const(np.log(init_prob / np.sum(init_prob))[None, :])
            elif prior_probs == 'tomask':  # absorb all states into the the mask type (last one)
                init_prob = 0.001 * np.ones(num_classes)
                init_prob[-1] = 1.
                self.prior_probs = to_torch_const(np.log(init_prob / np.sum(init_prob))[None, :])
            elif prior_probs == 'uniform':
                print('uniform prior types are used!')
                self.prior_probs = to_torch_const((np.ones(num_classes) / num_classes)[None, :])
            else:
                print('stat prior types are used!')
                log_probs = np.log(prior_probs.clip(min=1e-30))
                self.prior_probs = to_torch_const(log_probs)

    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t + self.prior_probs
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0) = alpha_bar_t * log_v0 + (1 - alpha_bar_t) * (prior_prob)
        # log_v0: (N, num_classes)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)
        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha + self.prior_probs
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

# Moldiff bond schedule    
def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0-A1)/(sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    # print(y)
    
    alphas_cumprod = y 
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod

def segment_schedule(timesteps, time_segment, segment_diff):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)
    
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    return betas