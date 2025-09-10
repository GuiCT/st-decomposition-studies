import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="PrintMatrices",
)
parser.add_argument("-f", "--file", required=True)
parser.add_argument("-hs", "--hash", nargs="+", required=True)
parser.add_argument("-o", "--output", default="matrices.txt")

if __name__ == "__main__":
    args = parser.parse_args()
    db = np.load(args.file)
    hashes = set(args.hash)
    np.set_printoptions(threshold=np.inf)
    text = ""

    for i, h in enumerate(hashes):
        if len(h) == 9:
            p1, p2 = h.split("...")
            h = next(k for k in db if k.startswith(p1) and k.endswith(p2))

        if h in db:
            p1 = h[:3]
            p2 = h[-3:]
            short_hash = f"{p1}...{p2}"
            A, S, T = db[h]
            full_A_str = np.array2string(A, precision=3, suppress_small=True, max_line_width=None)
            full_S_str = np.array2string(S, precision=3, suppress_small=True, max_line_width=None)
            full_T_str = np.array2string(T, precision=3, suppress_small=True, max_line_width=None)
            text += f"({short_hash}):\n"
            text += f"A =\n{full_A_str}\n\n"
            text += f"S =\n{full_S_str}\n\n"
            text += f"T =\n{full_T_str}\n\n"
        else:
            print(f"No matrix found with hash {h}.")
    with open(args.output, "w") as f:
        f.write(text)