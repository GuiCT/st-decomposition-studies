from argparse import ArgumentParser
from logging import Logger
import pandas as pd

import numpy as np

from st import check_for_minor_principals, st

parser = ArgumentParser(
    prog="DecompositionTester",
    description="Tests the ST decomposition for random integer A matrices",
)
parser.add_argument("-c", "--count", default=100)
parser.add_argument("-st", "--store", action="store_true")
parser.add_argument("-o", "--output")

if __name__ == "__main__":
    args = parser.parse_args()
    count = int(args.count)
    cols = {"A": [], "S": [], "T": [], "norm_frob": [], "norm_2": [], "norm_infinity": []}

    max_residuals = {
        "frob": -np.inf,
        "2": -np.inf,
        "infinity": -np.inf,
    }
    # Creating {{count}} different square matrices and testing the function
    for i in range(count):
        try:
            while True:
                n = 20
                A = np.random.rand(n, n) * 10 - 5
                if check_for_minor_principals(A):
                    S, T = st(A)
                    break
        except ValueError:
            continue
        print(f"Matrix {i+1}:")
        print("S:", S)
        print("T:", T)
        A_minus_ST = A - S @ T
        norm_frob = np.linalg.norm(A_minus_ST, ord="fro")
        norm_2 = np.linalg.norm(A_minus_ST, ord=2)
        norm_infty = np.linalg.norm(A_minus_ST, np.inf)
        print("||A - ST||F", norm_frob)
        print("||A - ST||2", norm_2)
        if args.store:
            cols["A"].append(A)
            cols["S"].append(S)
            cols["T"].append(T)
            cols["norm_frob"].append(norm_frob)
            cols["norm_2"].append(norm_2)
            cols["norm_infinity"].append(norm_infty)
        max_residuals["2"] = np.maximum(max_residuals["2"], norm_2)
        max_residuals["frob"] = np.maximum(max_residuals["frob"], norm_frob)
        max_residuals["infinity"] = np.maximum(max_residuals["infinity"], norm_infty)
    print("Max residual (Frobenius):", max_residuals["frob"])
    print("Max residual (2-order):", max_residuals["2"])
    print("Max residual (infinity-order):", max_residuals["infinity"])
    if args.store:
        res_df = pd.DataFrame(
            columns=["A", "S", "T", "||A - ST||_Frobenius", "||A - ST||_2", "||A - ST||_infinity"]
        )
        res_df["A"] = cols["A"]
        res_df["S"] = cols["S"]
        res_df["T"] = cols["T"]
        res_df["||A - ST||_Frobenius"] = cols["norm_frob"]
        res_df["||A - ST||_2"] = cols["norm_2"]
        res_df["||A - ST||_infinity"] = cols["norm_infinity"]
        out = "out.csv" if args.output is None else args.output
        res_df.to_csv(out, index=False)