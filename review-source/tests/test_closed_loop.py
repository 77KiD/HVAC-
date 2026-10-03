"""用假模擬器測閉環邏輯, 不需要 torch 也不需要訓練好的模型"""
import numpy as np
from hvac.closed_loop import run_episode, metrics
from hvac.controllers import FixedController


class DummySim:
    """控制量越大, 室溫往動作方向漂; 功率跟控制量成正比"""
    L = 12

    def __init__(self, spec):
        self.spec = spec

    def clip(self, a):
        return float(np.clip(a, *self.spec.act_range))

    def step(self, window, action, next_row):
        assert window.shape == (self.L, len(self.spec.seq_cols))
        assert next_row.shape == (len(self.spec.seq_cols),)
        T = window[-1, self.spec.room_idx]
        return T + 0.05, 10.0 + action


def _t0(spec, df):
    return int(np.argmax(df.timestamp.dt.month.isin(spec.test_months).to_numpy())) + DummySim.L


def test_fixed_controller_runs(ds):
    spec, dfs = ds
    df = next(iter(dfs.values()))
    hi = spec.act_range[1]
    ep = run_episode(DummySim(spec), FixedController(hi), df, _t0(spec, df), H=12)
    assert len(ep["T"]) == 12 and np.all(ep["action"] == hi)
    assert ep["T"][-1] > ep["T"][0]          # 狀態有被回饋進視窗
    m = metrics(ep, spec)
    assert set(m) == {"T_rmse", "comfort_violation_degh", "energy_kwh"}
    assert abs(m["energy_kwh"] - (10 + hi) * 12 * spec.dt_min / 60) < 1e-6


def test_action_is_clipped(ds):
    spec, dfs = ds
    df = next(iter(dfs.values()))
    ep = run_episode(DummySim(spec), FixedController(999), df, _t0(spec, df), H=3)
    assert np.all(ep["action"] == spec.act_range[1])
