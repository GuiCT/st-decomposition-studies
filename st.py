import numpy as np
import numpy.typing as npt


def check_for_minor_principals(A) -> bool:
    n = A.shape[0]
    for k in range(n):
        Ak = A[:k + 1, :k + 1]
        is_singular = np.linalg.det(Ak) == 0.0
        if is_singular:
            return False
    return True


def __st_base_case(A: npt.NDArray):
    a, b, c, d = A.flatten()
    a_negative = a < 0
    s_12 = -c if a_negative else c
    t_12num = -b - c * d if a_negative else -b + c * d
    t_12den = a + c ** 2 if a_negative else -a + c ** 2
    t_12 = t_12num / t_12den
    t_22 = d + c * t_12 if a_negative else d - c * t_12
    S = np.array([[np.abs(a), s_12], [s_12, 1.0]])
    T = np.array([[np.sign(a), t_12], [0.0, t_22]])
    return S, T


def st(A: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the symmetric-triangular decomposition of a square matrix A.

    The decomposition is such that A = S @ T, where S is symmetric and T is upper triangular.

    Parameters
    ----------
    A : npt.NDArray
        A square numpy array.
    Returns
    -------
    S : npt.NDArray
        A square symmetric matrix.
    T : npt.NDArray
        A square upper triangular matrix.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Input matrix A must be square.")
    
    if np.linalg.det(A) == 0:
        raise ValueError("Input matrix A must be non-singular.")

    S = np.zeros_like(A, dtype=np.float64)
    T = np.zeros_like(A, dtype=np.float64)
    L = np.zeros_like(A, dtype=np.float64)
    
    S[:2, :2], T[:2, :2] = __st_base_case(A[:2, :2])
    L[:2, :2] = np.linalg.cholesky(S[:2, :2])

    for k in range(1, n - 1):
        Lk = L[:k + 1, :k + 1]
        Tk = T[:k + 1, :k + 1]
        LkTprodinv = np.linalg.inv(Lk @ Lk.T)
        Lkinv = np.linalg.inv(Lk)
        Tkinv = np.linalg.inv(Tk)
        TkinvT = Tkinv.T
        lkp1 = Lkinv @ TkinvT @ A[k + 1, :k + 1]
        numerator = A[k + 1, k + 1] - A[k + 1, :k + 1] @ Tkinv @ LkTprodinv @ A[:k + 1, k + 1]
        beta = np.sign(numerator)
        tkp1 = LkTprodinv @ (A[:k + 1, k + 1] - beta * TkinvT @ A[k + 1, :k + 1])
        tau = np.sqrt(numerator / beta)
        L[k + 1, :k + 1] = lkp1
        L[k + 1, k + 1] = tau
        T[:k + 1, k + 1] = tkp1
        T[k + 1, k + 1] = beta
    
    return L @ L.T, T
