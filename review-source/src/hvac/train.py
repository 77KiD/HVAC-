"""
hvac.train — 訓練、預測、存讀檔、多步 rollout (CNN 組的核心函式庫)

scripts/train_cnn.py 與 scripts/evaluate_cnn.py 都只是呼叫這裡的函式。
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as hd
from .datasets import get_spec
from .models import ARCHS


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def build_model(arch, spec, L):
    return ARCHS[arch](len(spec.seq_cols), len(spec.now_cols), L)


# ───────────────────────── 訓練 ─────────────────────────
def _tensors(split, sc):
    xs, xn = sc.x(split["X_seq"], split["X_now"])
    yn = (split["y"] - sc.y_m) / sc.y_s
    return [torch.tensor(np.asarray(a, np.float32)) for a in (xs, xn, yn)]


def action_effect(model, xs, xn, sc, spec):
    """
    「持續改變控制量」對輸出的總效應 (物理單位): 對歷史視窗中所有控制量 + 下一步控制量的梯度加總。
    回傳 (dT/da, dP/da), 形狀 [B]
    """
    xs = xs.clone().requires_grad_(True)
    xn = xn.clone().requires_grad_(True)
    y = model(xs, xn)
    ai, an = spec.act_idx, spec.act_now_idx
    out = []
    for j in range(2):
        gs, gn = torch.autograd.grad(y[:, j].sum(), [xs, xn], create_graph=True)
        eff = gs[:, :, ai].sum(1) / float(sc.seq_s[ai]) + gn[:, an] / float(sc.now_s[an])
        out.append(eff * float(sc.y_s[j]))
    return out


def mono_penalty(model, xs, xn, sc, spec):
    """違反物理方向的梯度平方 (spec.mono_T / mono_P 給方向), 以標準化輸出單位計算"""
    dT, dP = action_effect(model, xs, xn, sc, spec)
    pen = 0.0
    if spec.mono_T:
        pen = pen + (torch.relu(-spec.mono_T * dT / float(sc.y_s[0])) ** 2).mean()
    if spec.mono_P:
        pen = pen + (torch.relu(-spec.mono_P * dP / float(sc.y_s[1])) ** 2).mean()
    return pen


def fit(arch, tr, va, sc, spec, L, seed=0, epochs=80, lr=1e-3, bs=256, patience=12,
        w_T=1.0, w_P=0.5, mono=None, verbose=True):
    """
    回傳 (最佳 val 的模型, history)。
    loss = 標準化後 ΔT 與 P 的加權 MSE + mono × 物理單調性懲罰 (mono 預設取 spec.mono_lambda)
    """
    mono = spec.mono_lambda if mono is None else mono
    set_seed(seed)
    model = build_model(arch, spec, L)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    w = torch.tensor([w_T, w_P])
    TR, VA = _tensors(tr, sc), _tensors(va, sc)
    hist = dict(train=[], val=[], val_T=[], val_P=[], best_epoch=0)
    best, best_state, bad, t_start = 1e9, None, 0, time.time()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(TR[0]))
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = (((model(TR[0][b], TR[1][b]) - TR[2][b]) ** 2) * w).mean()
            if mono > 0:
                loss = loss + mono * mono_penalty(model, TR[0][b], TR[1][b], sc, spec)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            e2 = (model(VA[0], VA[1]) - VA[2]) ** 2
            vl = (e2 * w).mean().item()
        sched.step(vl)
        hist["train"].append(tot / len(perm)); hist["val"].append(vl)
        hist["val_T"].append(e2[:, 0].mean().item()); hist["val_P"].append(e2[:, 1].mean().item())
        if vl < best - 1e-5:
            best, bad, hist["best_epoch"] = vl, 0, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and ep % 10 == 0:
            print(f"    [{arch} s{seed}] ep {ep:3d}  train {hist['train'][-1]:.4f}  val {vl:.4f}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    hist["seconds"] = time.time() - t_start
    hist["mono"] = mono
    return model, hist


def mono_violation(models, sc, spec, split, n=2000):
    """測試集上違反物理方向的樣本比例 (集成平均效應)"""
    if not isinstance(models, (list, tuple)):
        models = [models]
    idx = np.random.default_rng(0).choice(len(split["y"]), size=min(n, len(split["y"])), replace=False)
    xs, xn = sc.x(split["X_seq"][idx], split["X_now"][idx])
    xs, xn = torch.tensor(xs), torch.tensor(xn)
    dT = np.mean([action_effect(m, xs, xn, sc, spec)[0].detach().numpy() for m in models], 0)
    dP = np.mean([action_effect(m, xs, xn, sc, spec)[1].detach().numpy() for m in models], 0)
    out = {}
    if spec.mono_T:
        out["T_violation"] = float(np.mean(spec.mono_T * dT < 0))
        out["dT_da_median"] = float(np.median(dT))
    if spec.mono_P:
        out["P_violation"] = float(np.mean(spec.mono_P * dP < 0))
        out["dP_da_median"] = float(np.median(dP))
    return out


# ───────────────────────── 預測 ─────────────────────────
@torch.no_grad()
def predict_members(models, sc, X_seq, X_now):
    """每個成員的物理單位預測 → [M, N, 2]"""
    xs, xn = sc.x(X_seq, X_now)
    xs, xn = torch.tensor(xs), torch.tensor(xn)
    return np.stack([sc.y_inv(m(xs, xn).numpy()) for m in models])


def predict(models, sc, split):
    """集成平均 → [N, 2] (ΔT °C, P kW)"""
    if not isinstance(models, (list, tuple)):
        models = [models]
    return predict_members(models, sc, split["X_seq"], split["X_now"]).mean(0)


def one_step_metrics(pred, split):
    out = dict(T_rmse=rmse(pred[:, 0], split["y"][:, 0]),
               P_rmse=rmse(pred[:, 1], split["y"][:, 1]),
               T_persist=hd.persistence_rmse(split),
               P_std=float(split["y"][:, 1].std()))
    groups = list(dict.fromkeys(split["group"]))
    if len(groups) > 1:
        for g in groups:
            k = split["group"] == g
            out[f"T_rmse_{g}"] = rmse(pred[k, 0], split["y"][k, 0])
            out[f"P_rmse_{g}"] = rmse(pred[k, 1], split["y"][k, 1])
    return out


# ───────────────────────── 存讀檔 ─────────────────────────
def save_ckpt(path, arch, spec, L, model, sc, meta=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "dataset": spec.name, "L": L, "state": model.state_dict(),
                "scaler": sc.state_dict(), "meta": meta or {}}, path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    spec = get_spec(ck.get("dataset", "limassol"))
    m = build_model(ck["arch"], spec, ck["L"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, hd.Scaler().load_state_dict(ck["scaler"]), spec, ck


def load_ensemble(model_dir, arch="CNN_MLP"):
    """載入資料夾內同一架構的所有 seed → (models, scaler, spec, L)"""
    paths = sorted(Path(model_dir).glob(f"{arch.lower()}_s*.pt"))
    if not paths:
        raise FileNotFoundError(f"{model_dir} 沒有 {arch.lower()}_s*.pt, 先跑 scripts/train_cnn.py")
    loaded = [load_ckpt(p) for p in paths]
    _, sc, spec, ck = loaded[0]
    return [x[0] for x in loaded], sc, spec, ck["L"]


# ───────────────────────── 多步 rollout (批次化) ─────────────────────────
def rollout_starts(df, spec, L, H, months=None, stride=1):
    """在指定月份中, 找出 L 歷史 + H 未來都在同一連續段的起點 t0"""
    months = months or spec.test_months
    seg = df["seg"].to_numpy()
    mon = df.timestamp.dt.month.to_numpy()
    return np.array([t for t in range(L - 1, len(df) - H - 1, stride)
                     if mon[t + 1] in months and seg[t - L + 1] == seg[t + H]], dtype=int)


@torch.no_grad()
def rollout(models, sc, spec, df, t0s, H, L, act_override=None):
    """
    自我回饋 H 步: 室溫用模型預測填回視窗, 外生變數用真值。
    控制量: 預設用資料中的真值; 給 act_override 時, 歷史與未來的控制量都換成這個值 (掃描用)。
    回傳 dict: T, P, T_std (成員間標準差), T_true, P_true, 形狀 [N, H]
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    t0s = np.asarray(t0s, dtype=int)
    seq = df[spec.seq_cols].to_numpy(np.float32)
    now = df[spec.now_cols].to_numpy(np.float32)
    W = np.stack([seq[t - L + 1:t + 1] for t in t0s]).copy()
    if act_override is not None:
        W[:, :, spec.act_idx] = act_override
    T, P, S = [], [], []
    for k in range(H):
        idx = t0s + k + 1
        xn = now[idx].copy()
        if act_override is not None:
            xn[:, spec.act_now_idx] = act_override
        y = predict_members(models, sc, W, xn)                 # [M, N, 2]
        T_mem = W[None, :, -1, spec.room_idx] + y[..., 0]      # [M, N]
        T_next = T_mem.mean(0)
        row = seq[idx].copy()
        row[:, spec.room_idx] = T_next
        if act_override is not None:
            row[:, spec.act_idx] = act_override
        W = np.concatenate([W[:, 1:], row[:, None]], axis=1)
        T.append(T_next); P.append(y[..., 1].mean(0)); S.append(T_mem.std(0))
    Tt = df[spec.room_col].to_numpy(); Pt = df[spec.power_col].to_numpy()
    fut = t0s[:, None] + np.arange(1, H + 1)[None]
    return dict(T=np.stack(T, 1), P=np.stack(P, 1), T_std=np.stack(S, 1),
                T_true=Tt[fut], P_true=Pt[fut])


def persistence_rollout(spec, df, t0s, H):
    Tt = df[spec.room_col].to_numpy()
    t0s = np.asarray(t0s, dtype=int)
    fut = t0s[:, None] + np.arange(1, H + 1)[None]
    return dict(T=np.repeat(Tt[t0s][:, None], H, 1), T_true=Tt[fut])


def rollout_rmse_curve(models, sc, spec, dfs, L, H=None, groups=None, n_per=None, seed=0,
                       persistence=False):
    """各組隨機取起點 → 每個 horizon 的室溫 RMSE [H]"""
    H = H or spec.H
    n_per = n_per or (60 if len(dfs) > 1 else 300)
    rng = np.random.default_rng(seed)
    errs = []
    for g, df in dfs.items():
        if groups is not None and g not in groups:
            continue
        st = rollout_starts(df, spec, L, H)
        st = rng.choice(st, size=min(n_per, len(st)), replace=False)
        r = persistence_rollout(spec, df, st, H) if persistence else rollout(models, sc, spec, df, st, H, L)
        errs.append((r["T"] - r["T_true"]) ** 2)
    return np.sqrt(np.concatenate(errs).mean(0))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")
