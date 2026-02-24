import torch
from scipy.sparse.linalg import svds

@torch.no_grad()
def pinv(M: torch.Tensor, k: int = 2, tol: float = 1e-10, rtol:float = 1e-3, mode='randomized', power_iter: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute pseudo-inverse via truncated SVD. Memory-optimized version.
    Returns VhT, S_inv, U_T to avoid storing full pseudo-inverse matrix.
    """
    assert mode in ['randomized', 'scipy', 'torch', 'lobpcg'], "Invalid mode for pinv"
    with torch.no_grad():
        M = M.detach()
        if mode == 'torch':
            U, S, Vh = truncated_svd_torch(M,k=k,rtol=rtol)
        elif mode == 'randomized':
            U, S, Vh = randomized_SVD(M, k=k, rtol=rtol, q=power_iter)
        elif mode == 'scipy':
            U, S, Vh = truncated_svd_scipy(M,k=k,rtol=rtol)
        elif mode == 'lobpcg':
            U, S, Vh = truncated_svd_lobpcg(M, k=k, rtol=rtol)
        
        # threshold SVs to be above rtol * max SV
        kmax = 1 + (S > rtol * S[0]).nonzero(as_tuple=True)[0].max() # add 1 since answer is zero-indexed
        U = U[:,:kmax]
        S = S[:kmax]
        Vh = Vh[:kmax,:]

        # compute S_inv with tol threshold
        S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    
    return Vh.T.detach(), S_inv.detach(), U.T.detach()

@torch.no_grad()
def truncated_svd_lobpcg(A: torch.Tensor, k: int = 2, rtol:float=1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        else:
            C = A @ A.T       # (m x m)
            #X = torch.randn(m, k, device=A.device, dtype=A.dtype)
            evals, evecs = torch.lobpcg(C, k=k, largest=True)
            s = evals.clamp_min(0).sqrt()
            U = evecs
            Vh = ((U.T @ A) / s.clamp_min(torch.finfo(A.dtype).eps).reshape(-1,1)).conj()
        
        return U.detach(), s.detach(), Vh.detach()

@torch.no_grad()     
def truncated_svd_torch(A: torch.Tensor, k: int = 2, rtol:float=1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        A = A.detach()
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        U = U[:,:k]
        S = S[:k]
        Vh = Vh[:k,:]
        return U.detach(), S.detach(), Vh.detach()

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
    
    return U, S, Vh

def truncated_svd_scipy(A, k, rtol=1e-3):
    M = A.cpu().numpy()
    k = min(k, min(M.shape)-1)
    U, S, Vh = svds(M, k=int(k))
    
    # Sort in descending order (not guaranteed by svds)
    idx = S.argsort()[::-1]
    S = S[idx]
    U = U[:,idx]
    Vh = Vh[idx,:]

    # convert to torch and truncate
    U = torch.from_numpy(U).to(A.device, A.dtype)
    S = torch.from_numpy(S).to(A.device, A.dtype)
    Vh = torch.from_numpy(Vh).to(A.device, A.dtype)

    return U, S, Vh