"""
Implements the SpectralInv optimizer.

A single optimizer class that supports multiple spectral inversion strategies:
  - "svd":      Explicit SVD, sigma -> 1/sigma (pseudo or damped).
  - "ns":       Newton-Schulz (Denman-Beavers), sigma -> sigma/(sigma^2 + eps).
  - "cholesky": Damped Cholesky solve, sigma -> sigma/(sigma^2 + eps).
  - "qdwh":     QDWH/block-QR iteration, sigma -> sigma/(sigma^2 + eps).
  - "shifted":  Polar + NS inverse, sigma -> 1/(sigma + lambda).

1D parameters and embeddings are updated with Adam (no bias correction).
"""
import torch
import numpy as np

from . import EXCLUDED_LAYERS
from .spectral_inv_func import inv_sigvals


class SpectralInv(torch.optim.Optimizer):
    """
    SpectralInv - Gradient update with spectral inversion.

    Runs EMA momentum on gradients, applies a spectral inversion transform
    (singular values are inverted), and RMS-rescales the update.

    1D parameters and embeddings are updated with Adam (no bias correction).

    Arguments:
        named_params: Iterator of (name, param) pairs from model.named_parameters().
        lr: The learning rate. (1e-3 is a good default)
        spectral_lr_scale: Multiple for lr of spectrally inverted params.
        weight_decay: Weight decay coefficient.
        momentum: The momentum used for gradient EMA accumulation. (0.95 is a good default)
        betas: (beta1, beta2) for Adam on 1D params.
        eps: Epsilon for Adam on 1D params.
        target_rms: Target RMS for the update matrix. (0.18 is the default)
        strategy: Spectral inversion strategy.
            "svd": Explicit SVD inversion.
            "ns": Newton-Schulz (Denman-Beavers) iteration.
            "cholesky": Damped Cholesky solve.
            "qdwh": QDWH/block-QR iteration.
            "shifted": Polar + NS inverse.
        inv_strategy: For strategy="svd" only: "pseudo" or "damp".
        T: Number of iterations (for "ns" and "qdwh").
        damp_factor: Relative damping (for "ns").
        inv_eps: Ridge damping (for "cholesky", "qdwh", and "shifted").
        lower_bound: Min singular value estimate (for "qdwh").
        polar_steps: Newton-Schulz steps for polar factor (for "shifted").
        ns_inv_iters: Newton-Schulz iterations for matrix inverse (for "shifted").
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
                 inv_strategy="pseudo",
                 T=10,
                 damp_factor=1e-6,
                 inv_eps=1e-6,
                 lower_bound=1e-5,
                 polar_steps=5,
                 ns_inv_iters=10,
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
        self.inv_strategy = inv_strategy
        self.T = T
        self.damp_factor = damp_factor
        self.inv_eps = inv_eps
        self.lower_bound = lower_bound
        self.polar_steps = polar_steps
        self.ns_inv_iters = ns_inv_iters

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
            #      SpectralInv         #
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
                u = inv_sigvals(
                    buf_2d,
                    strategy=self.strategy,
                    inv_strategy=self.inv_strategy,
                    T=self.T,
                    damp_factor=self.damp_factor,
                    eps=self.inv_eps,
                    lower_bound=self.lower_bound,
                    polar_steps=self.polar_steps,
                    ns_inv_iters=self.ns_inv_iters,
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
