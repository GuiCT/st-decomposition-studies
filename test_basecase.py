from argparse import ArgumentParser

import numpy as np
import pandas as pd

from st import __st_base_case, check_for_minor_principals

parser = ArgumentParser(
    prog="DecompositionTester",
    description="Tests the ST decomposition for random integer A matrices",
)
parser.add_argument("-c", "--count", default=100)
parser.add_argument("-r", "--range", default=10)
parser.add_argument("-st", "--store", action="store_true")
parser.add_argument("-o", "--output", default="test_basecase.csv")

if __name__ == "__main__":
    args = parser.parse_args()
    count = int(args.count)
    random_range = float(args.range)
    cols = {"A": [], "L": [], "S": [], "T": [],
            "norm_frob": [], "norm_2": [], "norm_infinity": []}

    # Creating {{count}} different 2x2 matrices and testing the function
    for i in range(count):
        while True:
            A = np.random.rand(2, 2) * random_range - random_range / 2
            if check_for_minor_principals(A):
                break
        L, T = __st_base_case(A)
        print(f"Matrix {i+1}:")
        print("L:", L)
        print("T:", T)
        A_minus_LLT = A - L @ L.T @ T
        norm_frob = np.linalg.norm(A_minus_LLT, ord="fro")
        norm_2 = np.linalg.norm(A_minus_LLT, ord=2)
        norm_infty = np.linalg.norm(A_minus_LLT, np.inf)
        print("||A - LL^T T||F", norm_frob)
        print("||A - LL^T T||2", norm_2)
        print("||A - LL^T T||∞", norm_infty)
        cols["A"].append(A)
        cols["L"].append(L)
        cols["T"].append(T)
        cols["S"].append(L @ L.T)
        cols["norm_frob"].append(norm_frob)
        cols["norm_2"].append(norm_2)
        cols["norm_infinity"].append(norm_infty)
    max_residual = np.max(
        [cols["norm_frob"], cols["norm_2"], cols["norm_infinity"]])
    print("Max residual:", max_residual)
    if args.store:
        df = pd.DataFrame(cols)
        df.to_csv(args.output, index=False)
