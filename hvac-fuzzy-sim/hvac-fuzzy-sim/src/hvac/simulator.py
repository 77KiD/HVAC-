"""
hvac.simulator — 閉環用的模擬器介面 (CNN 組 ↔ 模糊/DE 組 的合約)

    sim = HVACSimulator.load("out/lbnl/models")        # 載入集成 (建議)
    T_next, P_next = sim.step(window, action, next_row)
    (T, P), (T_std, P_std) = sim.step_dist(window, action, next_row)

window   : np.ndarray [L, len(spec.seq_cols)]  過去 L 步 (欄位順序見 sim.spec.seq_cols)
action   : 下一步的控制量 (LBNL = 閥門開度 0–1, Limassol = 冷水 setpoint °C), 會被 clip 到 spec.act_range
next_row : np.ndarray [len(spec.seq_cols)]     下一步的資料列 (提供外生變數: 室外溫度、時間…)
介面若要改, 先在群組講, 再改這個檔和 tests/
"""
from pathlib import Path

import numpy as np

from .train import load_ckpt, load_ensemble, predict_members


class HVACSimulator:
    def __init__(self, models, scaler, spec, L):
        self.models = models if isinstance(models, (list, tuple)) else [models]
        self.sc, self.spec, self.L = scaler, spec, L

    @classmethod
    def load(cls, path, arch="CNN_MLP"):
        p = Path(path)
        if p.is_dir():
            return cls(*load_ensemble(p, arch))
        m, sc, spec, ck = load_ckpt(p)
        return cls(m, sc, spec, ck["L"])

    def clip(self, action):
        lo, hi = self.spec.act_range
        return float(np.clip(action, lo, hi))

    def _x_now(self, action, next_row):
        s = self.spec
        row = dict(zip(s.seq_cols, next_row))
        row[s.action_col] = self.clip(action)
        return np.array([[row[c] for c in s.now_cols]], np.float32)

    def _members(self, window, action, next_row):
        y = predict_members(self.models, self.sc, window[None].astype(np.float32),
                            self._x_now(action, next_row))[:, 0]
        return window[-1, self.spec.room_idx] + y[:, 0], y[:, 1]

    def step(self, window, action, next_row):
        T, P = self._members(window, action, next_row)
        return float(T.mean()), float(P.mean())

    def step_dist(self, window, action, next_row):
        T, P = self._members(window, action, next_row)
        return (float(T.mean()), float(P.mean())), (float(T.std()), float(P.std()))
