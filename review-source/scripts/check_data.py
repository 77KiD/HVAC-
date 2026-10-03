"""確認資料讀取與切分正常, 並印出 persistence 對照組數字

    python scripts/check_data.py                 # LBNL
    python scripts/check_data.py --dataset limassol
"""
import argparse
import numpy as np
from hvac import data as hd
from hvac.datasets import get_spec

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="lbnl", choices=["lbnl", "limassol"])
ap.add_argument("--data", default="data")
a = ap.parse_args()
spec = get_spec(a.dataset)
dfs = spec.load(a.data)
tr, va, te = hd.build_splits(dfs, spec)
print(f"[{spec.title}]")
print(f"train {len(tr['y'])}  val {len(va['y'])}  test {len(te['y'])}  X_seq {tr['X_seq'].shape}")
print(f"persistence 單步室溫 RMSE (test): {hd.persistence_rmse(te):.4f} °C")
if len(dfs) > 1:
    for g in dfs:
        m = te["group"] == g
        print(f"  {g}: {np.sqrt(np.mean(te['y'][m, 0] ** 2)):.4f}")
