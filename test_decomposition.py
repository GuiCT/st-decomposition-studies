from argparse import ArgumentParser
from collections import OrderedDict
from hashlib import sha1
from logging import Formatter, Logger, StreamHandler

import numpy as np
import pandas as pd

from st import check_for_minor_principals, st

parser = ArgumentParser(
    prog="DecompositionTester",
    description="Tests the ST decomposition for random integer A matrices",
)
parser.add_argument("-c", "--count", default=100)
output_res_args = parser.add_argument_group("results options")
output_res_args.add_argument("-or", "--output-results", default=None)
output_log_args = parser.add_argument_group("log options")
output_log_args.add_argument("-ol", "--output-log", default=None)
output_matrices_args = parser.add_argument_group("save matrices options")
output_matrices_args.add_argument("-om", "--output-matrices", default=None)
decomposition_settings_args = parser.add_argument_group(
    "decomposition settings")
decomposition_settings_args.add_argument("-s", "--size", default=10)
decomposition_settings_args.add_argument("-t", "--tolerance", default=0.1)

logger = Logger("DecompositionTester", level="INFO")
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.output_log is not None:
        file_handler = StreamHandler(
            open(args.output_log, "a", encoding="utf-8"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    count = int(args.count)
    n = int(args.size)
    tol = float(args.tolerance)
    norms = {
        "frob": "fro",
        "2": 2,
        "infinity": np.inf
    }

    cols = {"i": [], "A_hash": []}
    for norm in norms:
        cols[f"norm_{norm}"] = []
    matrices_dict = OrderedDict[str, np.ndarray]()

    # Creating {{count}} different square matrices and testing the function
    for i in range(count):
        try:
            while True:
                A = np.random.rand(n, n) * 10 - 5
                if check_for_minor_principals(A, tol):
                    S, T = st(A)
                    break
        except ValueError as e:
            logger.error(f"Matrix {i+1} could not be decomposed")
            logger.error(e)
            logger.error(A)
            continue
        A_hash = sha1(A.data).hexdigest()
        if args.output_matrices is not None:
            triple = np.zeros((3, *A.shape))
            triple[0, :, :] = A
            triple[1, :, :] = S
            triple[2, :, :] = T
            matrices_dict[A_hash] = triple
        logger.info(
            f"========== Matrix {i+1} ({A_hash[:3]}...{A_hash[-3:]}) ==========")
        A_minus_ST = A - S @ T
        for norm in norms:
            cols[f"norm_{norm}"].append(
                np.linalg.norm(A_minus_ST, ord=norms[norm]))
            logger.info(f"||A - ST||{norm}: {cols[f'norm_{norm}'][-1]}")
        cols["i"].append(i)
        cols["A_hash"].append(A_hash)
    logger.info("====================")
    logger.info("========== STATS ==========")
    max_residuals = {k: np.argmax(cols[f"norm_{k}"]) for k in norms}
    mean_residuals = {k: np.mean(cols[f"norm_{k}"]) for k in norms}
    logger.info("========== WORST CASE SCENARIOS ==========")
    for k in norms:
        max_hash = cols["A_hash"][max_residuals[k]]
        logger.info(
            f"Max residual (||.||_{k}, {max_hash[:3]}...{max_hash[-3:]}): {cols[f'norm_{k}'][max_residuals[k]]}")
    logger.info("========== AVERAGE CASE SCENARIOS ==========")
    for k in norms:
        logger.info(f"Mean residual (||.||_{k}): {mean_residuals[k]}")
    logger.info("====================")

    if args.output_results is not None:
        res_df = pd.DataFrame(cols)
        res_df.to_csv(args.output_results, index=False)

    if args.output_matrices is not None:
        np.savez_compressed(args.output_matrices, **matrices_dict) # type: ignore
