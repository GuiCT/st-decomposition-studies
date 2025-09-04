import numpy as np

def check_singularity(A):
    """Check if a matrix is singular."""
    return np.linalg.cond(A) > 1 / np.finfo(A.dtype).eps

if __name__ == "__main__":
    from st import __st_base_case

    max_residual = -np.inf
    # Creating 100 different 2x2 matrices and testing the function
    for i in range(100):
        while True:
            A = np.random.rand(2, 2) * 10 - 5
            if not check_singularity(A):
                break
        S, T = __st_base_case(A)
        print(f"Matrix {i+1}:")
        print("S:", S)
        print("T:", T)
        print("||A - ST||:", np.linalg.norm(A - S @ T))
        max_residual = np.maximum(max_residual, np.linalg.norm(A - S @ T))
    print("Max residual:", max_residual)