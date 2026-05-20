"""
Implements the Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

Adapted from: https://github.com/mtcrawshaw/muon-and-friends/blob/master/src/muon_and_friends/muonadam.py
Original Muon: https://github.com/KellerJordan/Muon/blob/master/muon.py
"""
import numpy as np
import torch

from . import EXCLUDED_LAYERS
from .polar_express import PolarExpress



def SVDPolarFactorBottomP(G: torch.Tensor, p: float):
    """Polar factor keeping only the bottom p% of singular vectors.

    Parameters
    ----------
    G : 2D tensor
    p : float in (0, 1]. Fraction of singular vectors to keep (smallest SVs).
        E.g. p=0.9 keeps the 90% of directions with the smallest singular values.
    """
    assert G.ndim == 2, "Input tensor must have exactly two dimensions."
    assert 0.0 < p <= 1.0, "p must be in (0, 1]."

    X = G.clone()
    if G.size(-2) > G.size(-1):
        X = X.mT

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    # Keep bottom p fraction: S is descending, so keep the last k entries.
    k = max(1, int(np.ceil(p * len(S))))
    # Polar factor of the truncated part: U @ Vh
    X = U[:, len(S) - k:] @ Vh[len(S) - k:]

    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X.bfloat16()

    return X



class MuonBottomP(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs EMA momentum on gradients, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use PolarExpress, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    1D parameters and embeddings are updated with Adam (no bias correction).

    Arguments:
        named_params: Iterator of (name, param) pairs from model.named_parameters().
        lr: The learning rate. (1e-3 is a good default)
        muon_lr_scale: Multiple for lr of Muon params. (10 is a good default)
        weight_decay: Weight decay coefficient.
        momentum: The momentum used for gradient EMA accumulation. (0.95 is a good default)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        betas: (beta1, beta2) for Adam on 1D params.
        eps: Epsilon for Adam on 1D params.
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 muon_lr_scale=1, 
                 weight_decay=0.1,
                 momentum=0.95,
                 ns_steps=5,
                 betas=(0.95, 0.95),
                 eps=1e-8,
                 target_rms=0.18,
                 only_train_matrices=False,
                 bottom_p=0.9
                 ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            eps=eps,
        )

        self.muon_lr_scale = muon_lr_scale
        self.ns_steps = ns_steps
        self.target_rms = target_rms
        self.bottom_p = bottom_p

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
            #           Muon           #
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
                u = SVDPolarFactorBottomP(buf_2d, self.bottom_p).view(buf.shape)
                rms = u.norm() / np.sqrt(u.numel())
                rms_scale = self.target_rms / rms.clamp(min=1e-12)
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(u, alpha=-lr * self.muon_lr_scale * rms_scale)

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
