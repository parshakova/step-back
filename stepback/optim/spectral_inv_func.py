"""
Spectral transform functions for gradient updates.

All functions take a 2D gradient matrix G and return a transformed matrix
of the same shape. They differ in how they map the singular values.

Inversion strategies:
  - svd_inv:      Explicit SVD, sigma -> 1/sigma (pseudo or damped).
  - ns_inv:       Newton-Schulz (Denman-Beavers), sigma -> sigma/(sigma^2 + eps).
  - cholesky_inv: Damped Cholesky solve, sigma -> sigma/(sigma^2 + eps).
  - qdwh_inv:     QDWH/block-QR iteration, sigma -> sigma/(sigma^2 + eps).
  - shifted_spectral_inv: Polar + Cholesky, sigma -> 1/(sigma + lambda).

References:
  - "Muon is Not That Special" (arXiv:2605.11181) for Freon/QDWH.
"""
import math
import torch
from itertools import repeat
from typing import Optional


# ===========================================================================
# SVD-based spectral inversion
# ===========================================================================

def svd_inv(G: torch.Tensor, inv_strategy: str = "pseudo") -> torch.Tensor:
    """
    Spectral inversion via explicit SVD.

    Args:
        G: 2D gradient matrix.
        inv_strategy: "pseudo" (hard cutoff at eps) or "damp" (soft 1/(sigma+eps)).

    Returns:
        Spectrally inverted tensor, same shape as G.
    """
    assert G.ndim == 2, "Input tensor must have exactly two dimensions."

    norm = G.norm() + 1e-12
    X = G.clone() / norm

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    if inv_strategy == "pseudo":
        tol = torch.finfo(G.dtype).eps
        S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    elif inv_strategy == "damp":
        eps = torch.finfo(G.dtype).eps
        S_inv = 1.0 / (S + eps)
    else:
        raise ValueError(f"Unknown inv_strategy: {inv_strategy!r}. Use 'pseudo' or 'damp'.")
    X = U @ (S_inv[:, None] * Vh) / norm

    return X.bfloat16()


# ===========================================================================
# Newton-Schulz (Denman-Beavers) spectral inversion
# ===========================================================================

def _matrix_inv_sqrt_NS_core(Ahat: torch.Tensor, T: int = 10):
    """
    Coupled Denman-Beavers iteration for Ahat^{-1/2}.
    Eigenvalues of Ahat must be in (0, 1].
    """
    device = Ahat.device
    n = Ahat.size(-1)

    I = torch.eye(n, dtype=Ahat.dtype, device=device)
    Y = Ahat.clone()
    Z = I.expand_as(Ahat).clone()

    for _ in range(T):
        T_mat = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T_mat
        Z = T_mat @ Z

    return Z


def _matrix_inv_sqrt_NS(A: torch.Tensor, alpha, T: int = 10):
    """Compute A^{-1/2} via Denman-Beavers with eigenvalue scaling."""
    assert A.size(-2) == A.size(-1), "A must be square in last two dimensions"

    Ahat = A / alpha
    Z = _matrix_inv_sqrt_NS_core(Ahat, T)

    if isinstance(alpha, torch.Tensor):
        return Z / torch.sqrt(alpha)
    else:
        return Z / math.sqrt(alpha)


def ns_inv(
    G: torch.Tensor,
    T: int = 10,
    damp_factor: float = 1e-6,
) -> torch.Tensor:
    """
    Spectral inversion via coupled Denman-Beavers (Newton-Schulz) iteration.

    Computes (XX^T + eps*I)^{-1} X, mapping sigma -> sigma/(sigma^2 + eps).
    Obtains (XX^T)^{-1/2} via NS iteration, then squares.

    Args:
        G: 2D gradient matrix.
        T: Number of Newton-Schulz iterations.
        damp_factor: Relative damping (adds damp_factor * s_max * I).

    Returns:
        Spectrally inverted tensor, same shape as G.
    """
    if G.ndim == 1:
        norm = G.norm()
        eps = torch.finfo(G.dtype).eps
        return G / (norm**2 + eps)

    out_dtype = G.dtype
    compute_dtype = torch.float32 if G.dtype in (torch.float16, torch.bfloat16) else G.dtype
    X = G.detach().to(compute_dtype)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m = X.size(-2)
    device = X.device
    dtype = X.dtype
    tiny = torch.finfo(dtype).eps

    nu = X.norm(p="fro").clamp_min(tiny)
    Xn = X / nu

    XnXnT = Xn @ Xn.mT
    XnXnT = 0.5 * (XnXnT + XnXnT.mT)

    gershgorin_bound = XnXnT.abs().sum(dim=-1).max()
    frobenius_bound = XnXnT.norm()
    s_max = torch.minimum(gershgorin_bound, frobenius_bound)

    I = torch.eye(m, dtype=dtype, device=device)
    A = XnXnT + damp_factor * s_max * I
    alpha = s_max * (1.0 + damp_factor)

    InvSqrt = _matrix_inv_sqrt_NS(A, alpha, T=T)
    Ainv = InvSqrt @ InvSqrt
    out = (Ainv @ Xn) / nu

    if transposed:
        out = out.mT

    return out.to(out_dtype)


# ===========================================================================
# Damped Cholesky solve (Freon c=1 direction)
# ===========================================================================

@torch.no_grad()
def cholesky_inv(
    G: torch.Tensor,
    eps: float = 1e-6,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Spectral inversion via damped Cholesky solve.

    Frobenius-normalizes G, solves the damped normal equations via Cholesky.
    Spectrally: sigma -> sigma / (sigma^2 + eps * ||G||_F^2).

    Based on "Muon is Not That Special" (arXiv:2605.11181).

    Args:
        G: 2D gradient matrix.
        eps: Ridge damping (relative to Frobenius normalization).
        compute_dtype: Internal dtype. Defaults to float32 for half types.

    Returns:
        Spectrally inverted tensor, same shape and dtype as G.
    """
    out_dtype = G.dtype
    if compute_dtype is None:
        compute_dtype = torch.float32 if G.dtype in (torch.float16, torch.bfloat16) else G.dtype

    X = G.detach().to(compute_dtype)

    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    m, n = X.shape
    device = X.device
    dtype = X.dtype
    tiny = torch.finfo(dtype).eps

    nu = X.norm(p="fro").clamp_min(tiny)
    Xn = X / nu

    I = torch.eye(m, device=device, dtype=dtype)
    A = Xn @ Xn.T
    A = 0.5 * (A + A.T)
    eps = max(eps, tiny)
    ridge = torch.tensor(float(eps), device=device, dtype=dtype)
    Dn = None
    for _ in range(8):
        L, info = torch.linalg.cholesky_ex(A + ridge * I)
        if int(info.max()) == 0:
            Dn = torch.cholesky_solve(Xn, L)
            break
        ridge = 10.0 * ridge + tiny

    if Dn is None:
        Dn = torch.linalg.solve(A + ridge * I, Xn)

    D = Dn / nu

    if transposed:
        D = D.T

    return D.to(out_dtype)


# ===========================================================================
# QDWH / block-QR (Algorithm 4 style) spectral inversion
# ===========================================================================

def _qdwh_polar_coeffs(num_iters: int = 5, lower_bound: float = 1e-5):
    """
    QDWH coefficients for polar / inverse-square-root iteration.

    Scalar map: s_{t+1} = s_t * (a_t + b_t s_t^2) / (1 + c_t s_t^2).
    """
    l = float(lower_bound)
    coeffs = []

    for _ in range(num_iters):
        l = min(max(l, 1e-30), 1.0)
        l2 = l * l

        d = (4.0 * (1.0 - l2) / (l2 * l2)) ** (1.0 / 3.0)
        sqrt_d = math.sqrt(1.0 + d)

        a = sqrt_d + 0.5 * math.sqrt(
            max(0.0, 8.0 - 4.0 * d + 8.0 * (2.0 - l2) / (l2 * sqrt_d))
        )
        b = ((a - 1.0) ** 2) / 4.0
        c = a + b - 1.0

        coeffs.append((a, b, c))
        l = l * (a + b * l2) / (1.0 + c * l2)

    return coeffs


@torch.no_grad()
def qdwh_inv(
    G: torch.Tensor,
    eps: float = 1e-12,
    T: int = 5,
    lower_bound: float = 1e-5,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Spectral inversion via QDWH/block-QR iteration.

    Uses QDWH polar coefficients for the inverse-square-root subproblem,
    accumulates (XX^T)^{-1/2} via coupled Cholesky/QR steps, then squares
    to obtain (XX^T)^{-1}.

    Based on Algorithm 4 from "Muon is Not That Special" (arXiv:2605.11181).

    Args:
        G: 2D gradient matrix.
        eps: Ridge damping for augmented QR.
        T: Number of QDWH iterations.
        lower_bound: Lower bound on min singular value after normalization.
        compute_dtype: Internal dtype. Defaults to float32 for half types.

    Returns:
        Spectrally inverted tensor, same shape and dtype as G.
    """
    out_dtype = G.dtype
    if compute_dtype is None:
        compute_dtype = torch.float32 if G.dtype in (torch.float16, torch.bfloat16) else G.dtype

    X = G.detach().to(compute_dtype)

    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    m, n = X.shape
    device = X.device
    dtype = X.dtype
    tiny = torch.finfo(dtype).eps

    nu = X.norm(p="fro").clamp_min(tiny)
    Xn = X / nu

    I = torch.eye(m, device=device, dtype=dtype)

    aug = torch.cat([Xn.T, math.sqrt(float(eps)) * I], dim=0)
    _, R = torch.linalg.qr(aug, mode="reduced")
    L = R.T.contiguous()

    C = I.clone()

    for alpha_f, beta_f, gamma_f in _qdwh_polar_coeffs(T, lower_bound):
        alpha = torch.tensor(alpha_f, device=device, dtype=dtype)
        beta = torch.tensor(beta_f, device=device, dtype=dtype)
        gamma = torch.tensor(gamma_f, device=device, dtype=dtype)

        rho = beta / gamma

        if gamma_f <= 1.0:
            K = torch.cat([torch.sqrt(gamma) * L.T, I], dim=0)
        else:
            K = torch.cat([L.T, torch.sqrt(1.0 / gamma) * I], dim=0)

        Q, _ = torch.linalg.qr(K, mode="reduced")
        Q2 = Q[-m:, :]
        V = Q2 @ Q2.T

        W = rho * I + (alpha - rho) * V
        W = 0.5 * (W + W.T)

        L = W @ L
        C = W @ C

    D = (C @ C @ Xn) / nu

    if transposed:
        D = D.T

    return D.to(out_dtype)


# ===========================================================================
# Shifted spectral inverse (Polar + Cholesky)
# ===========================================================================

# PolarExpress coefficients (same as polar.py, but without @torch.compile
# so this module works on machines where the Triton/Inductor compiler is
# unavailable).
_POLAR_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
_POLAR_COEFFS = [(a / 1.01, b / 1.01**3, c / 1.01**5)
                  for a, b, c in _POLAR_COEFFS[:-1]] + [_POLAR_COEFFS[-1]]


def _polar_factor(G: torch.Tensor, steps: int = 5, frob_eps: float = 1e-2,
                  deflation_eps: float = 1e-2) -> torch.Tensor:
    """Approximate polar factor P ≈ UV^T via Newton-Schulz (same as PolarExpress)."""
    X = G.bfloat16()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + frob_eps))
    hs = _POLAR_COEFFS[:steps] + list(repeat(_POLAR_COEFFS[-1], max(0, steps - len(_POLAR_COEFFS))))
    for a, b, c in hs:
        a = a / (1 + deflation_eps)
        b = b / (1 + deflation_eps)
        c = c / (1 + deflation_eps)
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _newton_schulz_inverse(A: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """
    Approximate A^{-1} via Newton-Schulz iteration:

        X_{k+1} = X_k (2I - A X_k)

    A must be square, nonsingular. Initial guess X_0 = alpha * A^T
    with alpha = 1 / ||A||_F^2 (guarantees convergence when all
    singular values of A are < 1/alpha).
    """
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    tiny = torch.finfo(A.dtype).eps

    alpha = 1.0 / (A.norm(p="fro") ** 2).clamp_min(tiny)
    X = alpha * A.T

    for _ in range(num_iters):
        X = X @ (2.0 * I - A @ X)

    return X


@torch.no_grad()
def shifted_spectral_inv(
    G: torch.Tensor,
    eps: float = 1e-6,
    polar_steps: int = 5,
    ns_inv_iters: int = 15,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Shifted spectral inverse via polar factor + Newton-Schulz inverse.

    Given G = U S V^T, computes U diag(1/(S + lambda)) V^T.

    Uses Newton-Schulz (PolarExpress) to get P ≈ U V^T, forms the
    symmetric H = G P^T ≈ U S U^T, then approximates (H + lambda I)^{-1}
    via Newton-Schulz iteration and multiplies by P.

    Args:
        G: 2D gradient matrix.
        eps: Damping (relative to Frobenius norm).
        polar_steps: Number of Newton-Schulz steps for polar factor.
        ns_inv_iters: Number of Newton-Schulz iterations for matrix inverse.
        compute_dtype: Internal dtype. Defaults to float32 for half types.

    Returns:
        Shifted-inverted tensor, same shape and dtype as G.
    """
    assert G.ndim == 2

    out_dtype = G.dtype
    if compute_dtype is None:
        compute_dtype = torch.float32 if G.dtype in (torch.float16, torch.bfloat16) else G.dtype

    X = G.detach().to(compute_dtype)

    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T.contiguous()
        transposed = True

    m, n = X.shape
    device = X.device
    dtype = X.dtype
    tiny = torch.finfo(dtype).eps

    nu = X.norm(p="fro").clamp_min(tiny)
    Xn = X / nu

    # Polar factor: P ≈ U V^T  (computed in bf16 internally)
    P = _polar_factor(Xn, steps=polar_steps).to(dtype)

    # H = Xn @ P^T ≈ U S U^T   (m x m, symmetric)
    H = Xn @ P.T
    H = 0.5 * (H + H.T)

    # A = H + delta * I, then Y = A^{-1} P ≈ U diag(1/(s + delta)) V^T
    delta = max(float(eps), tiny)
    I = torch.eye(m, device=device, dtype=dtype)
    A = H + delta * I
    A = 0.5 * (A + A.T)

    Ainv = _newton_schulz_inverse(A, num_iters=ns_inv_iters)
    Y = Ainv @ P

    D = Y / nu

    if transposed:
        D = D.T

    return D.to(out_dtype)


# ===========================================================================
# Unified dispatch
# ===========================================================================

def inv_sigvals(
    G: torch.Tensor,
    strategy: str = "cholesky",
    inv_strategy: str = "pseudo",
    T: int = 10,
    damp_factor: float = 1e-6,
    eps: float = 1e-6,
    lower_bound: float = 1e-5,
    polar_steps: int = 5,
    ns_inv_iters: int = 10,
) -> torch.Tensor:
    """
    Spectral inversion: maps singular values sigma -> ~1/sigma (damped).

    Args:
        G: Input matrix (2D tensor).
        strategy: "svd", "ns", "cholesky", "qdwh", or "shifted".
        inv_strategy: For strategy="svd" only: "pseudo" or "damp".
        T: Number of iterations (for "ns" and "qdwh").
        damp_factor: Relative damping (for "ns").
        eps: Ridge damping (for "cholesky", "qdwh", and "shifted").
        lower_bound: Min singular value estimate (for "qdwh").
        polar_steps: Newton-Schulz steps for polar factor (for "shifted").
        ns_inv_iters: Newton-Schulz iterations for matrix inverse (for "shifted").

    Returns:
        Spectrally inverted tensor with same shape as G.
    """
    if strategy == "svd":
        return svd_inv(G, inv_strategy=inv_strategy)
    elif strategy == "ns":
        return ns_inv(G, T=T, damp_factor=damp_factor)
    elif strategy == "cholesky":
        return cholesky_inv(G, eps=eps)
    elif strategy == "qdwh":
        return qdwh_inv(G, eps=eps, T=T, lower_bound=lower_bound)
    elif strategy == "shifted":
        return shifted_spectral_inv(G, eps=eps, polar_steps=polar_steps, ns_inv_iters=ns_inv_iters)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'svd', 'ns', 'cholesky', 'qdwh', or 'shifted'.")
