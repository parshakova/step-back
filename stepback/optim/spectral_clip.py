"""
Implements the SpectralClip optimizer.

A single optimizer class that supports multiple spectral clipping strategies:
  - "svd": Hard clamp via explicit SVD, sigma -> min(sigma, tau).
  - "ns":  Soft clip via Newton-Schulz, sigma -> sigma / sqrt(1 + sigma^2 / c^2).

1D parameters and embeddings are updated with Adam (no bias correction).
"""
import torch
import numpy as np

from . import EXCLUDED_LAYERS
from .spectral_clip_func import clip_sigvals


class SpectralClip(torch.optim.Optimizer):
    """
    SpectralClip - Gradient update with spectral clipping.

    Runs EMA momentum on gradients, applies a spectral clipping transform
    (large singular values are capped), and RMS-rescales the update.

    1D parameters and embeddings are updated with Adam (no bias correction).

    Arguments:
        named_params: Iterator of (name, param) pairs from model.named_parameters().
        lr: The learning rate. (1e-3 is a good default)
        spectral_lr_scale: Multiple for lr of spectral-clipped params. (10 is a good default)
        weight_decay: Weight decay coefficient.
        momentum: The momentum used for gradient EMA accumulation. (0.95 is a good default)
        betas: (beta1, beta2) for Adam on 1D params.
        eps: Epsilon for Adam on 1D params.
        target_rms: Target RMS for the update matrix. (0.18 is the default)
        strategy: Spectral clipping strategy.
            "svd": Hard clamp via explicit SVD.
            "ns": Soft clip via Newton-Schulz iterations.
        tau_strategy: How to set the clipping threshold.
            "one": Fixed threshold (tau=1 for SVD, clip_c for NS).
            "average": Data-dependent (mean singular value for SVD, Frobenius est. for NS).
        clip_c: Base clipping threshold (for strategy="ns" with tau_strategy="one").
        ns_iter: Number of Newton-Schulz iterations (for strategy="ns").
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 spectral_lr_scale=1,
                 weight_decay=0.1,
                 momentum=0.95,
                 betas=(0.95, 0.95),
                 eps=1e-8,
                 target_rms=0.18,
                 only_train_matrices=False,
                 strategy="svd",
                 tau_strategy="average",
                 clip_c=1.0,
                 ns_iter=10,
                 ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            eps=eps,
        )
        self.target_rms = target_rms
        self.spectral_lr_scale = spectral_lr_scale
        self.strategy = strategy
        self.tau_strategy = tau_strategy
        self.clip_c = clip_c
        self.ns_iter = ns_iter

        muon_params = []
        adam_params = []
        for name, p in named_params:
            if p.ndim >= 2 and not any(ex in name for ex in EXCLUDED_LAYERS):
                muon_params.append(p)
            elif not only_train_matrices:
                adam_params.append(p)

        params = list(muon_params) + list(adam_params)
        super().__init__(params, defaults)

        for p in muon_params:
            self.state[p]["use_muon"] = True

        for p in adam_params:
            self.state[p]["use_muon"] = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            ############################
            #      SpectralClip        #
            ############################
            for p in group["params"]:
                state = self.state[p]
                g = p.grad
                if not state["use_muon"] or g is None:
                    continue

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1.0 - momentum)

                buf_2d = buf.view(buf.shape[0], -1)
                u = clip_sigvals(
                    buf_2d,
                    strategy=self.strategy,
                    tau_strategy=self.tau_strategy,
                    clip_c=self.clip_c,
                    ns_iter=self.ns_iter,
                ).view(buf.shape)

                # RMS rescale to target_rms
                rms = u.norm() / np.sqrt(u.numel())
                rms_scale = self.target_rms / rms.clamp(min=1e-12)

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(u, alpha=-lr * self.spectral_lr_scale * rms_scale)

            ############################
            #       Adam backup        #
            ############################
            for p in group["params"]:
                state = self.state[p]
                g = p.grad
                if state["use_muon"] or g is None:
                    continue

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                    state["sq_momentum_buffer"] = g.square()
                buf = state["momentum_buffer"]
                buf2 = state["sq_momentum_buffer"]
                buf.lerp_(g, 1.0 - beta1)
                buf2.lerp_(g.square(), 1.0 - beta2)

                g = buf / (eps + buf2.sqrt())
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr)

        return loss
