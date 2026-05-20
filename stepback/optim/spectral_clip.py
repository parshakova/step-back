"""
Implements the SpectralClip optimizer.

Uses SVD-based spectral clipping instead of polar factorization for the
orthogonalization step. Singular values are clamped to a maximum of `tau`,
so with tau=1 this recovers the polar factor (all singular values set to 1),
and with larger tau it allows the update to retain magnitude information.

1D parameters and embeddings are updated with Adam (no bias correction).
"""
import torch
import numpy as np

from . import EXCLUDED_LAYERS


def SVDClip(G: torch.Tensor, tau_strategy: str) -> torch.Tensor:
    assert G.ndim == 2, "Input tensor must have exactly two dimensions."

    X = G.clone()  # torch doesn't support SVD for bfloat16

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    if tau_strategy == "one":
        tau = 1.0
    elif tau_strategy == "average":
        tau = S.sum() / min(G.shape[-2], G.shape[-1])
    clipped = torch.clamp(S, max=tau)
    X = U @ (clipped[:, None] * Vh)

    X = X.bfloat16()
    return X


class SpectralClip(torch.optim.Optimizer):
    """
    SpectralClip - Gradient update with SVD-based spectral clipping.

    Like Muon, this runs EMA momentum on gradients, but replaces the polar
    factorization step with SVD clipping: singular values are clamped to at
    most `tau`, preserving partial magnitude information in the update.

    1D parameters and embeddings are updated with Adam (no bias correction).

    Arguments:
        named_params: Iterator of (name, param) pairs from model.named_parameters().
        lr: The learning rate. (1e-3 is a good default)
        spectral_lr_scale: Multiple for lr of spectral-clipped params. (10 is a good default)
        weight_decay: Weight decay coefficient.
        momentum: The momentum used for gradient EMA accumulation. (0.95 is a good default)
        betas: (beta1, beta2) for Adam on 1D params.
        eps: Epsilon for Adam on 1D params.
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
                 tau_strategy="average"
                 ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            eps=eps,
        )
        self.target_rms = target_rms
        self.tau_strategy = tau_strategy
        self.spectral_lr_scale = spectral_lr_scale

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
                u = SVDClip(buf_2d, self.tau_strategy).view(buf.shape)

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
