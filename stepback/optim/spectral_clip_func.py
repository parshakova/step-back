"""
Spectral clipping functions for gradient updates.

All functions take a 2D gradient matrix G and return a clipped matrix
of the same shape. They differ in how they clip the singular values.

Clipping strategies:
  - svd_clip:  Explicit SVD, hard clamp sigma -> min(sigma, tau).
  - ns_clip:   Newton-Schulz soft clip, sigma -> sigma / sqrt(1 + sigma^2 / c^2).

Threshold (tau) strategies:
  - "one":     Fixed threshold (tau=1.0 for SVD, clip_c for NS).
  - "average": Data-dependent threshold (mean singular value for SVD,
               Frobenius estimate for NS).

References:
  - NS soft clip adapted from https://github.com/mlolab/llm-spectral-clipping
    (commit ed9df11d797aa94f312f4d7810d5f8cd8ddc9897)
"""
import math
import torch


# ===========================================================================
# SVD-based hard spectral clipping
# ===========================================================================

def svd_clip(G: torch.Tensor, tau_strategy: str = "average") -> torch.Tensor:
    """
    Hard spectral clipping via explicit SVD.

    Singular values are clamped to at most tau:
      - tau_strategy="one":     tau = 1.0
      - tau_strategy="average": tau = sum(S) / min(m, n)

    Args:
        G: 2D gradient matrix.
        tau_strategy: "one" or "average".

    Returns:
        Spectrally clipped tensor, same shape as G (bfloat16).
    """
    assert G.ndim == 2, "Input tensor must have exactly two dimensions."

    X = G.clone()

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    if tau_strategy == "one":
        tau = 1.0
    elif tau_strategy == "average":
        tau = S.sum() / min(G.shape[-2], G.shape[-1])
    else:
        raise ValueError(f"Unknown tau_strategy: {tau_strategy!r}. Use 'one' or 'average'.")
    clipped = torch.clamp(S, max=tau)
    X = U @ (clipped[:, None] * Vh)

    X = X.bfloat16()
    return X


# ===========================================================================
# Newton-Schulz soft spectral clipping
# ===========================================================================

def _matrix_inv_sqrt_NS_core(Ahat: torch.Tensor, ns_iter: int = 10):
    """
    Coupled Denman-Beavers iteration for Ahat^{-1/2}.
    Eigenvalues of Ahat must be in (0, 1].
    """
    device = Ahat.device
    n = Ahat.size(-1)

    I = torch.eye(n, dtype=Ahat.dtype, device=device)
    Y = Ahat.clone()
    Z = I.expand_as(Ahat).clone()

    for _ in range(ns_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    return Z


def _matrix_inv_sqrt_NS(A: torch.Tensor, alpha, ns_iter: int = 10):
    """Compute A^{-1/2} via Denman-Beavers with eigenvalue scaling."""
    assert A.size(-2) == A.size(-1), "A must be square in last two dimensions"

    Ahat = A / alpha
    Z = _matrix_inv_sqrt_NS_core(Ahat, ns_iter)

    if isinstance(alpha, torch.Tensor):
        return Z / torch.sqrt(alpha)
    else:
        return Z / math.sqrt(alpha)


def ns_clip(
    G: torch.Tensor,
    clip_c: float = 1.0,
    ns_iter: int = 10,
    tau_strategy: str = "average",
) -> torch.Tensor:
    """
    Soft spectral clipping via Newton-Schulz iterations.

    Computes Hc(X) = (I + XX^T / c^2)^{-1/2} @ X, which maps
    sigma -> sigma / sqrt(1 + sigma^2 / c^2).

    Adapted from https://github.com/mlolab/llm-spectral-clipping

    Args:
        G: 2D gradient matrix.
        clip_c: Base clipping threshold (used when tau_strategy="one").
        ns_iter: Number of Newton-Schulz iterations.
        tau_strategy: How to set the clipping threshold.
            "one": use clip_c directly.
            "average": estimate mean singular value via Frobenius norm.

    Returns:
        Spectrally clipped tensor, same shape and dtype as G.
    """
    if tau_strategy == "average":
        clip_c = G.norm().item() / min(G.shape[-2], G.shape[-1])

    if G.ndim == 1:
        norm = G.norm()
        if norm <= clip_c:
            return G
        return G / torch.sqrt(1 + (norm / clip_c) ** 2)

    X = G.bfloat16()

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m = X.size(-2)
    XXT = X @ X.mT

    gershgorin_bound = XXT.abs().sum(dim=-1).max()
    frobenius_bound = XXT.norm()
    s_max_XXT = torch.minimum(gershgorin_bound, frobenius_bound)

    I = torch.eye(m, dtype=X.dtype, device=X.device)
    A = I + XXT / (clip_c ** 2)
    alpha = 1.0 + s_max_XXT / (clip_c ** 2)

    InvSqrt = _matrix_inv_sqrt_NS(A, alpha, ns_iter=ns_iter)
    out = InvSqrt @ X

    if transposed:
        out = out.mT

    return out.to(G.dtype)


# ===========================================================================
# Unified dispatch
# ===========================================================================

def clip_sigvals(
    G: torch.Tensor,
    strategy: str = "ns",
    tau_strategy: str = "average",
    clip_c: float = 1.0,
    ns_iter: int = 10,
) -> torch.Tensor:
    """
    Spectral clipping: caps large singular values.

    Args:
        G: Input matrix (2D tensor).
        strategy: "svd" (hard clamp) or "ns" (soft clip via Newton-Schulz).
        tau_strategy: "one" (fixed threshold) or "average" (data-dependent).
        clip_c: Base clipping threshold (for strategy="ns" with tau_strategy="one").
        ns_iter: Number of Newton-Schulz iterations (for strategy="ns").

    Returns:
        Spectrally clipped tensor with same shape as G.
    """
    if strategy == "svd":
        return svd_clip(G, tau_strategy=tau_strategy)
    elif strategy == "ns":
        return ns_clip(G, clip_c=clip_c, ns_iter=ns_iter, tau_strategy=tau_strategy)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'svd' or 'ns'.")
