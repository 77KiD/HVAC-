"""Day 3 範例: 用 CNN 集成模擬器跑固定控制量 baseline (測試月份)

    python scripts/run_closed_loop.py                     # LBNL
    python scripts/run_closed_loop.py --dataset limassol
"""
import argparse
import numpy as np
from hvac.simulator import HVACSimulator
from hvac.controllers import FixedController
from hvac.closed_loop import run_episode, metrics
from hvac.train import rollout_starts

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="lbnl", choices=["lbnl", "limassol"])
ap.add_argument("--data", default="data")
ap.add_argument("--model", default=None, help="預設 out/<dataset>/models (集成)")
a = ap.parse_args()

sim = HVACSimulator.load(a.model or f"out/{a.dataset}/models")
spec = sim.spec
dfs = spec.load(a.data)
df = dfs[16] if len(dfs) > 1 else list(dfs.values())[0]
H = spec.long_H
t0 = rollout_starts(df, spec, sim.L, H)[0]
print(f"[{spec.title}] 模擬器: {len(sim.models)} 個模型集成, L={sim.L}, 模擬 {spec.steps_to_h(H):g} h, "
      f"起點 {df.timestamp.iloc[t0 + 1]}")
lo, hi = spec.act_range
for v in np.linspace(lo, hi, 5):
    ep = run_episode(sim, FixedController(float(v)), df, t0, H=H)
    print(f"固定 {spec.action_col} = {v:5.2f}  ", {k: round(x, 3) for k, x in metrics(ep, spec).items()})
