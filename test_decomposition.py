import numpy as np

def check_singularity(A):
    """Check if a matrix is singular."""
    return np.linalg.cond(A) > 1 / np.finfo(A.dtype).eps

if __name__ == "__main__":
    from st import st

    max_residual = -np.inf
    # Creating 100 different square matrices and testing the function
    for i in range(100):
        try:
            while True:
                # n = np.random.randint(10, 100)
                n = 3
                A = np.random.rand(n, n) * 10 - 5
                if not check_singularity(A):
                    S, T = st(A)
                    break
        except ValueError:
            continue
        print(f"Matrix {i+1}:")
        print("S:", S)
        print("T:", T)
        print("||A - ST||:", np.linalg.norm(A - S @ T))
        max_residual = np.maximum(max_residual, np.linalg.norm(A - S @ T))
    print("Max residual:", max_residual)