"""
hvac.closed_loop — 控制器 → 模擬器 → 下一步狀態 (負責人: Day 3)

外生變數 (天氣、時間、外氣風門) 用資料真值, 室溫由模擬器自己滾動產生。
sim 只要有 .spec / .L / .step() / .clip() 就能用 (HVACSimulator 或測試用假模擬器)。
"""
import numpy as np


def run_episode(sim, controller, df, t0, H, T_target=None):
    spec, L = sim.spec, sim.L
    T_target = spec.comfort_target if T_target is None else T_target
    seq = df[spec.seq_cols].to_numpy(np.float32)
    window = seq[t0 - L + 1:t0 + 1].copy()
    act = float(window[-1, spec.act_idx])
    e_prev = window[-1, spec.room_idx] - T_target
    out_T, out_P, out_a = [], [], []

    for k in range(H):
        t = t0 + k
        e = window[-1, spec.room_idx] - T_target
        act = sim.clip(controller(e, e - e_prev, act))
        e_prev = e
        T_next, P_next = sim.step(window, act, seq[t + 1])
        row = seq[t + 1].copy()
        row[spec.room_idx], row[spec.act_idx] = T_next, act
        window = np.vstack([window[1:], row])
        out_T.append(T_next); out_P.append(P_next); out_a.append(act)

    return dict(T=np.array(out_T), P=np.array(out_P), action=np.array(out_a),
                timestamp=df.timestamp.iloc[t0 + 1:t0 + 1 + H].to_numpy(), dt_min=spec.dt_min)


def metrics(ep, spec):
    """Day 4/5 用: 溫度誤差、舒適帶違反 (°C·h)、能耗 (kWh)"""
    T, hrs = ep["T"], ep["dt_min"] / 60
    lo, hi = spec.comfort_band
    viol = np.clip(lo - T, 0, None) + np.clip(T - hi, 0, None)
    return dict(T_rmse=float(np.sqrt(np.mean((T - spec.comfort_target) ** 2))),
                comfort_violation_degh=float(viol.sum() * hrs),
                energy_kwh=float(ep["P"].sum() * hrs))
