import torch

@torch.no_grad()
def pinv(M: torch.Tensor, k: int = 2, tol: float = 1e-10, rtol:float = 1e-3, full=False, randomized=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute pseudo-inverse via truncated SVD. Memory-optimized version.
    Returns VhT, S_inv, U_T to avoid storing full pseudo-inverse matrix.
    """
    with torch.no_grad():
        M = M.detach()
        if full:
            U, S, Vh = truncated_svd_full(M,k=k,rtol=rtol)
        elif randomized:
            U, S, Vh = randomized_SVD(M, k=k, rtol=rtol)
        else:
            U, S, Vh = truncated_svd(M, k=k, rtol=rtol)
        S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    return Vh.T.detach(), S_inv.detach(), U.T.detach()

@torch.no_grad()
def truncated_svd(A: torch.Tensor, k: int = 2, rtol:float=1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        A = A.detach()
        m, n = A.shape
        if n <= m:
            C = A.T @ A       # (n x n), symmetric PSD
            # Use LOBPCG or plain eigen; eigsh-style not yet standard in torch
            # torch.lobpcg expects symmetric; give random init
            #X = torch.randn(n, k, device=A.device, dtype=A.dtype)
            evals, evecs = torch.lobpcg(C, k=k, largest=True)  # top-k
            s = evals.clamp_min(0).sqrt()
            V = evecs
            # Left sing vecs: U = A V / s
            U = (A @ V) / s.clamp_min(torch.finfo(A.dtype).eps)
            Vh = V.T
            #s = torch.where(s > rtol * s[0], s, torch.zeros_like(s))

            # truncate with rtol
            kmax = (s > rtol * s[0]).nonzero(as_tuple=True)[0].max()
            U = U[:,:kmax]
            s = s[:kmax]
            Vh = Vh[:kmax,:]
            return U.detach(), s.detach(), Vh.detach()
        else:
            C = A @ A.T       # (m x m)
            #X = torch.randn(m, k, device=A.device, dtype=A.dtype)
            evals, evecs = torch.lobpcg(C, k=k, largest=True)
            s = evals.clamp_min(0).sqrt()
            U = evecs
            Vh = ((U.T @ A) / s.clamp_min(torch.finfo(A.dtype).eps).reshape(-1,1)).conj()

            #s = torch.where(s > rtol * s[0], s, torch.zeros_like(s))

            # truncate with rtol
            kmax = (s > rtol * s[0]).nonzero(as_tuple=True)[0].max()
            U = U[:,:kmax]
            s = s[:kmax]
            Vh = Vh[:kmax,:]
            return U.detach(), s.detach(), Vh.detach()

@torch.compile
@torch.no_grad()     
def truncated_svd_full(A: torch.Tensor, k: int = 2, rtol:float=1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        A = A.detach()
        U, S, Vh = torch.linalg.svd(A, full_matrices=True)
        U = U[:,:k]
        S = S[:k]
        Vh = Vh[:k,:]
        S = torch.where(S > rtol * S[0], S, torch.zeros_like(S))
        return U.detach(), S.detach(), Vh.detach()

@torch.compile
@torch.no_grad()
def randomized_SVD(A, k, p=5, q=1, rtol:float=1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD algorithm. Memory-efficient for rank-k approximations.
    p: oversampling parameter (default 5)
    q: power iteration parameter (default 1)
    """
    m,n = A.shape
    r = min(k+p, min(m, n))  # Don't exceed matrix dimensions
    device = A.device
    dtype = A.dtype

    # Random projection
    Omega = torch.randn(n, r, device=device, dtype=dtype)
    Y = A @ Omega
    
    # Power iterations for better accuracy
    for _ in range(q):
        Y = A @ (A.T @ Y)
    
    # QR decomposition
    Q, _ = torch.linalg.qr(Y)
    
    # Project and compute SVD of smaller matrix
    B = Q.T @ A
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    # Truncate to rank k
    U = U[:,:k].contiguous()
    S = S[:k].contiguous()
    Vh = Vh[:k,:].contiguous()
    
    # Threshold small singular values
    kmax = (S > rtol * S[0]).nonzero(as_tuple=True)[0].max()
    U = U[:,:kmax]
    S = S[:kmax]
    Vh = Vh[:kmax,:]
    
    return U, S, Vh