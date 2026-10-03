"""
train_cnn.py — Day 2 主實驗: 1D-CNN+MLP vs MLP-only vs persistence, 多個 seed

    python scripts/train_cnn.py                       # LBNL (預設), 5 seeds
    python scripts/train_cnn.py --dataset limassol    # 備案資料
    python scripts/train_cnn.py --quick               # 快速測試: 1 seed, 15 epochs

輸出 (out/<dataset>/):
    models/cnn_mlp_s{k}.pt, mlponly_s{k}.pt   CNN 集成即模擬器 (HVACSimulator.load("out/lbnl/models"))
    results/main.json                         所有數字
    figures/fig1_loss.png             訓練/驗證 loss
    figures/fig2_pred_vs_actual.png   ★ Day 2 完成標準: 預測 vs 實際
    figures/fig3_scatter.png          預測 vs 實際散佈圖
    figures/fig4_rollout.png          ★ Day 2 完成標準: 多步 rollout 誤差
"""
import argparse
from pathlib import Path

import numpy as np

from hvac import data as hd
from hvac import train as T
from hvac.datasets import get_spec
from hvac.plotting import setup, COLORS

ARCH_LIST = ["CNN_MLP", "MLPOnly"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lbnl", choices=["lbnl", "limassol"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None, help="預設 out/<dataset>")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--mono", type=float, default=None, help="物理單調性約束權重 (預設依資料集: LBNL 1.0)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs = 1, 15
    spec = get_spec(a.dataset)
    L, H = a.L or spec.L, spec.H
    mstr = "、".join(map(str, spec.test_months)) + " 月"
    out = Path(a.out or f"out/{spec.name}"); fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    dfs = spec.load(a.data)
    tr, va, te = hd.build_splits(dfs, spec, L)
    sc = hd.Scaler().fit(tr)
    print(f"[{spec.title}] train {len(tr['y'])} / val {len(va['y'])} / test {len(te['y'])}   L={L} 步")

    res = dict(dataset=spec.name, title=spec.title,
               config=dict(L=L, H=H, dt_min=spec.dt_min, seeds=a.seeds, epochs=a.epochs,
                           n_train=len(tr["y"]), n_val=len(va["y"]), n_test=len(te["y"]),
                           train_months=spec.train_months, val_months=spec.val_months,
                           test_months=spec.test_months,
                           mono=spec.mono_lambda if a.mono is None else a.mono),
               persistence=dict(T_rmse=hd.persistence_rmse(te)), runs={}, summary={})
    models, hists = {k: [] for k in ARCH_LIST}, {k: [] for k in ARCH_LIST}

    for arch in ARCH_LIST:
        res["runs"][arch] = []
        for s in range(a.seeds):
            print(f"== {arch} seed {s}")
            m, h = T.fit(arch, tr, va, sc, spec, L, seed=s, epochs=a.epochs, mono=a.mono)
            T.save_ckpt(out / "models" / f"{arch.lower()}_s{s}.pt", arch, spec, L, m, sc,
                        meta=dict(seed=s, best_epoch=h["best_epoch"]))
            met = T.one_step_metrics(T.predict(m, sc, te), te)
            met["rollout"] = T.rollout_rmse_curve(m, sc, spec, dfs, L, H)
            met["history"] = h
            res["runs"][arch].append(met)
            models[arch].append(m); hists[arch].append(h)
            print(f"   T {met['T_rmse']:.4f}  P {met['P_rmse']:.2f}  roll@end {met['rollout'][-1]:.3f}")

    for arch in ARCH_LIST:
        runs = res["runs"][arch]
        keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
        summ = {k: dict(mean=float(np.mean([r[k] for r in runs])),
                        std=float(np.std([r[k] for r in runs]))) for k in keys}
        roll = np.array([r["rollout"] for r in runs])
        summ["rollout_mean"], summ["rollout_std"] = roll.mean(0), roll.std(0)
        ens = T.one_step_metrics(T.predict(models[arch], sc, te), te)
        ens["rollout"] = T.rollout_rmse_curve(models[arch], sc, spec, dfs, L, H)
        ens.update(T.mono_violation(models[arch], sc, spec, te))
        summ["ensemble"] = ens
        res["summary"][arch] = summ
    res["persistence"]["rollout"] = T.rollout_rmse_curve(None, sc, spec, dfs, L, H, persistence=True)
    T.save_json(res, out / "results" / "main.json")

    # ─── 圖 ───
    plt = setup()
    step_lab = f"步 ({spec.dt_min} 分鐘)"

    # fig1 loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for arch in ARCH_LIST:
        for i, h in enumerate(hists[arch]):
            ax[0].plot(h["train"], color=COLORS[arch], alpha=.35, lw=1)
            ax[0].plot(h["val"], color=COLORS[arch], lw=1.5, label=arch if i == 0 else None)
            ax[0].axvline(h["best_epoch"], color=COLORS[arch], ls=":", lw=.8, alpha=.5)
            ax[1].plot(h["val_T"], color=COLORS[arch], lw=1.2, label=f"{arch} ΔT" if i == 0 else None)
            ax[1].plot(h["val_P"], color=COLORS[arch], lw=1.2, ls="--", label=f"{arch} P" if i == 0 else None)
    ax[0].set(title="Loss（淡線 = train，實線 = val，點線 = best epoch）", xlabel="epoch", ylabel="weighted MSE", yscale="log")
    ax[1].set(title="Validation loss 分項（標準化）", xlabel="epoch", yscale="log")
    ax[0].legend(); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_loss.png", dpi=140); plt.close(fig)

    # fig2 pred vs actual: 測試集中一段 (多組時取最後一組)
    g_show = list(dfs)[-1]
    k = np.where(te["group"] == g_show)[0]
    k = k[:min(len(k), 168 if spec.dt_min == 60 else 6 * 192)]
    preds = {arch: T.predict(models[arch], sc, te) for arch in ARCH_LIST}
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4, 3]))
    xi = np.arange(len(k))
    T_act = te["T_now"][k] + te["y"][k, 0]
    ax[0].plot(xi, T_act, color=COLORS["actual"], lw=1.3, label="實際")
    ax[2].plot(xi, te["y"][k, 1], color=COLORS["actual"], lw=1.3, label="實際")
    for arch in ARCH_LIST:
        p = preds[arch]
        ax[0].plot(xi, te["T_now"][k] + p[k, 0], color=COLORS[arch], lw=1, alpha=.85, label=f"{arch}（集成）")
        ax[1].plot(xi, te["T_now"][k] + p[k, 0] - T_act, color=COLORS[arch], lw=.8, alpha=.85,
                   label=f"{arch}  RMSE {T.rmse(p[k, 0], te['y'][k, 0]):.3f}")
        ax[2].plot(xi, p[k, 1], color=COLORS[arch], lw=1, alpha=.85, label=f"{arch}（集成）")
    ax[1].plot(xi, -te["y"][k, 0], color=COLORS["persistence"], lw=.6, alpha=.6,
               label=f"persistence  RMSE {T.rmse(0, te['y'][k, 0]):.3f}")
    ax[1].axhline(0, color="#EEEEEE", lw=.6)
    grp = f"，{spec.group_label} {g_show}" if len(dfs) > 1 else ""
    ax[0].set(ylabel="室溫 (°C)", title=f"測試集單步預測 vs 實際（{mstr}{grp}）")
    ax[1].set(ylabel="室溫誤差 (°C)")
    ax[2].set(ylabel="HVAC 功率 (kW)", xlabel=f"樣本序號（每步 {spec.dt_min} 分鐘）")
    for x in ax:
        x.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_pred_vs_actual.png", dpi=140); plt.close(fig)

    # fig3 scatter
    p = preds["CNN_MLP"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    Tt, Tp = te["T_now"] + te["y"][:, 0], te["T_now"] + p[:, 0]
    color = te["X_now"][:, spec.act_now_idx]
    for axi, yt, yp, lab in [(ax[0], Tt, Tp, "室溫 (°C)"), (ax[1], te["y"][:, 1], p[:, 1], "HVAC 功率 (kW)")]:
        sca = axi.scatter(yt, yp, c=color, cmap="cool", s=3, alpha=.4)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        axi.plot([lo, hi], [lo, hi], color="#EEEEEE", lw=1, ls="--")
        r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
        axi.set(xlabel=f"實際 {lab}", ylabel=f"預測 {lab}", title=f"CNN_MLP 集成   R² = {r2:.3f}")
    fig.colorbar(sca, ax=ax, label=spec.act_label)
    fig.savefig(fig_dir / "fig3_scatter.png", dpi=140); plt.close(fig)

    # fig4 rollout
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hh = np.arange(1, H + 1) * spec.dt_min
    for arch in ARCH_LIST:
        mu, sd = res["summary"][arch]["rollout_mean"], res["summary"][arch]["rollout_std"]
        ax.plot(hh, mu, color=COLORS[arch], marker="o", ms=3, label=f"{arch}（mean ± std，{a.seeds} seeds）")
        ax.fill_between(hh, mu - sd, mu + sd, color=COLORS[arch], alpha=.2)
    ax.plot(hh, res["persistence"]["rollout"], color=COLORS["persistence"], ls="--", label="persistence")
    ax.set(xlabel="預測時間長度（分鐘）", ylabel="室溫 RMSE (°C)",
           title=f"多步 rollout 誤差（{mstr}測試集，控制量用實際紀錄）")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_rollout.png", dpi=140); plt.close(fig)

    print("\n=== 主實驗摘要 ===")
    pr = res["persistence"]
    print(f"persistence      T {pr['T_rmse']:.4f}              roll@{hh[-1]}min {pr['rollout'][-1]:.3f}")
    for arch in ARCH_LIST:
        s = res["summary"][arch]; e = s["ensemble"]
        print(f"{arch:9s} seeds  T {s['T_rmse']['mean']:.4f}±{s['T_rmse']['std']:.4f}  "
              f"P {s['P_rmse']['mean']:.2f}±{s['P_rmse']['std']:.2f}  roll {s['rollout_mean'][-1]:.3f}")
        print(f"{arch:9s} ens.   T {e['T_rmse']:.4f}          P {e['P_rmse']:.2f}        roll {e['rollout'][-1]:.3f}"
              + (f"   物理違反率 T {e['T_violation']:.1%}" if "T_violation" in e else ""))


if __name__ == "__main__":
    main()
