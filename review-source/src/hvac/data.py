"""
hvac.data — 視窗切分、時間切分、標準化 (資料集無關, 欄位都由 Spec 決定)
"""
import numpy as np

from .datasets import get_spec, Spec  # noqa: F401  (re-export)


def make_windows(df, spec, L):
    """
    對每個 t 產生一筆樣本:
      X_seq : [L, C]  t-L+1 .. t 的歷史
      X_now : [D]     t+1 的控制量 + 外生變數
      y     : [2]     (T[t+1]-T[t], P[t+1])
    """
    seq = df[spec.seq_cols].to_numpy(np.float32)
    now = df[spec.now_cols].to_numpy(np.float32)
    T = df[spec.room_col].to_numpy(np.float32)
    P = df[spec.power_col].to_numpy(np.float32)
    seg = df["seg"].to_numpy()
    month = df.timestamp.dt.month.to_numpy()
    idx = np.array([t for t in range(L - 1, len(df) - 1) if seg[t - L + 1] == seg[t + 1]], dtype=int)
    return dict(X_seq=np.stack([seq[t - L + 1:t + 1] for t in idx]),
                X_now=now[idx + 1],
                y=np.stack([T[idx + 1] - T[idx], P[idx + 1]], axis=1),
                T_now=T[idx], month=month[idx + 1],
                group=np.array([df["group"].iloc[0]] * len(idx), dtype=object), t_idx=idx)


def build_splits(dfs, spec, L=None):
    """依「月份」做時間切分 (不能隨機切, 否則相鄰時間點洩漏)"""
    L = L or spec.L
    parts = [make_windows(d, spec, L) for d in dfs.values()]
    allw = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def pick(months):
        m = np.isin(allw["month"], months)
        return {k: v[m] for k, v in allw.items()}

    return pick(spec.train_months), pick(spec.val_months), pick(spec.test_months)


def filter_group(split, groups):
    m = np.isin(split["group"], list(groups))
    return {k: v[m] for k, v in split.items()}


class Scaler:
    """只用 train 算 mean/std"""
    def fit(self, train):
        C = train["X_seq"].shape[-1]
        self.seq_m = train["X_seq"].reshape(-1, C).mean(0)
        self.seq_s = train["X_seq"].reshape(-1, C).std(0) + 1e-6
        self.now_m, self.now_s = train["X_now"].mean(0), train["X_now"].std(0) + 1e-6
        self.y_m, self.y_s = train["y"].mean(0), train["y"].std(0) + 1e-6
        return self

    def x(self, X_seq, X_now):
        return ((X_seq - self.seq_m) / self.seq_s).astype(np.float32), \
               ((X_now - self.now_m) / self.now_s).astype(np.float32)

    def y_inv(self, y_norm):
        return y_norm * self.y_s + self.y_m

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, d):
        self.__dict__.update(d)
        return self


def persistence_rmse(split):
    """最笨的對照組: 下一刻室溫 = 現在室溫 (ΔT=0)"""
    return float(np.sqrt(np.mean(split["y"][:, 0] ** 2)))
