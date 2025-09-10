import numpy as np
import numpy.typing as npt


def check_for_minor_principals(A, tol: float = 0.1) -> bool:
    n = A.shape[0]
    for k in range(n):
        Ak = A[:k + 1, :k + 1]
        is_singular = np.abs(np.linalg.det(Ak)) < tol
        if is_singular:
            return False
    return True


def __st_base_case(A: npt.NDArray):
    a, b, c, _ = A.flatten()

    L = np.zeros_like(A, dtype=np.float64)
    T = np.zeros_like(A, dtype=np.float64)

    detA = np.linalg.det(A)
    beta = np.sign(detA / a)
    tau = np.sqrt(np.abs(detA / a))
    t12 = b - beta * c / a

    L[0, 0] = 1.0
    L[1, 0] = c / a
    L[1, 1] = tau
    T[0, 0] = a
    T[0, 1] = t12
    T[1, 1] = beta

    return L, T


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

    L = np.zeros_like(A, dtype=np.float64)
    T = np.zeros_like(A, dtype=np.float64)

    L[:2, :2], T[:2, :2] = __st_base_case(A[:2, :2])

    for k in range(1, n - 1):
        Lk = L[:k + 1, :k + 1]
        Tk = T[:k + 1, :k + 1]
        LkTprodinv = np.linalg.inv(Lk @ Lk.T)
        Lkinv = np.linalg.inv(Lk)
        Tkinv = np.linalg.inv(Tk)
        TkinvT = Tkinv.T
        lkp1 = Lkinv @ TkinvT @ A[k + 1, :k + 1]
        numerator = A[k + 1, k + 1] - A[k + 1, :k +
                                        1] @ Tkinv @ LkTprodinv @ A[:k + 1, k + 1]
        beta = np.sign(numerator)
        tkp1 = LkTprodinv @ (A[:k + 1, k + 1] - beta *
                             TkinvT @ A[k + 1, :k + 1])
        tau = np.sqrt(numerator / beta)
        L[k + 1, :k + 1] = lkp1
        L[k + 1, k + 1] = tau
        T[:k + 1, k + 1] = tkp1
        T[k + 1, k + 1] = beta

    return L @ L.T, T


def st_stable(A: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
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

    L = np.zeros_like(A, dtype=np.float64)
    Linv = np.zeros_like(A, dtype=np.float64)
    T = np.zeros_like(A, dtype=np.float64)

    L[:2, :2], T[:2, :2] = __st_base_case(A[:2, :2])

    for k in range(1, n - 1):
        Lk = L[:k + 1, :k + 1]
        Tk = T[:k + 1, :k + 1]
        Lkinv = Linv[:k + 1, :k + 1]
        LkTprodinv = np.linalg.inv(Lk @ Lk.T)
        Tkinv = np.linalg.inv(Tk)
        TkinvT = Tkinv.T
        lkp1 = Lkinv @ TkinvT @ A[k + 1, :k + 1]
        numerator = A[k + 1, k + 1] - A[k + 1, :k +
                                        1] @ Tkinv @ LkTprodinv @ A[:k + 1, k + 1]
        beta = np.sign(numerator)
        tkp1 = LkTprodinv @ (A[:k + 1, k + 1] - beta *
                             TkinvT @ A[k + 1, :k + 1])
        tau = np.sqrt(numerator / beta)
        L[k + 1, :k + 1] = lkp1
        L[k + 1, k + 1] = tau
        T[:k + 1, k + 1] = tkp1
        T[k + 1, k + 1] = beta

    return L @ L.T, T
