"""
evaluate_cnn.py — 進階評估 (需先跑 train_cnn.py)

    python scripts/evaluate_cnn.py                     # LBNL (預設)
    python scripts/evaluate_cnn.py --dataset limassol
    python scripts/evaluate_cnn.py --quick

實驗:
  A. (只有 Limassol) 留出 setpoint: 只用 12/16/20°C 訓練, 測 14/18°C
  B. 控制量響應掃描: 控制量固定在網格上各值, rollout 後取平均 → 模型學到的「控制量 → 室溫/功率」關係
  C. 視窗長度消融: L = 12 / 24 / 48 步
  D. 集成不確定度: 長時間 rollout 的 mean ± 2σ vs 實際
  E. (有設定單調約束的資料集) 物理約束消融: 有約束 vs 無約束
輸出: out/<dataset>/results/eval.json, figures/fig5~fig8
"""
import argparse
from pathlib import Path

import numpy as np

from hvac import data as hd
from hvac import train as T
from hvac.plotting import setup, COLORS

HOLD_TRAIN, HOLD_TEST = (12, 16, 20), (14, 18)


def exp_holdout(dfs, spec, ens, sc_all, L, seeds, epochs):
    tr, va, te = hd.build_splits(dfs, spec, L)
    tr_h, va_h, te_h = hd.filter_group(tr, HOLD_TRAIN), hd.filter_group(va, HOLD_TRAIN), hd.filter_group(te, HOLD_TEST)
    sc = hd.Scaler().fit(tr_h)
    out = dict(train_groups=HOLD_TRAIN, test_groups=HOLD_TEST, holdout=[])
    models = []
    for s in range(seeds):
        print(f"== holdout seed {s}")
        m, _ = T.fit("CNN_MLP", tr_h, va_h, sc, spec, L, seed=s, epochs=epochs, verbose=False)
        models.append(m)
        met = T.one_step_metrics(T.predict(m, sc, te_h), te_h)
        met["rollout"] = T.rollout_rmse_curve(m, sc, spec, dfs, L, groups=HOLD_TEST)
        out["holdout"].append(met)
    out["seen"] = T.one_step_metrics(T.predict(ens, sc_all, te_h), te_h)
    out["seen"]["rollout"] = T.rollout_rmse_curve(ens, sc_all, spec, dfs, L, groups=HOLD_TEST)
    out["persistence"] = dict(T_rmse=hd.persistence_rmse(te_h),
                              rollout=T.rollout_rmse_curve(None, None, spec, dfs, L, groups=HOLD_TEST, persistence=True))
    return out, (models, sc)


def action_sweep(models, sc, spec, df, L):
    """控制量固定為網格值, 跑 sweep_H 步, 取 burn-in 之後的平均"""
    st = T.rollout_starts(df, spec, L, spec.sweep_H, stride=max(1, spec.sweep_H // 2))
    res = dict(grid=list(spec.sweep_grid), T=[], P=[], T_band=[], P_band=[])
    for a in spec.sweep_grid:
        r = T.rollout(models, sc, spec, df, st, spec.sweep_H, L, act_override=float(a))
        Tm, Pm = r["T"][:, spec.sweep_burn:].mean(1), r["P"][:, spec.sweep_burn:].mean(1)
        res["T"].append(float(Tm.mean())); res["P"].append(float(Pm.mean()))
        res["T_band"].append(float(Tm.std())); res["P_band"].append(float(Pm.std()))
    return res


def truth_by_group(dfs, spec):
    t, p = [], []
    for g, df in dfs.items():
        m = df.timestamp.dt.month.isin(spec.test_months)
        t.append(float(df.loc[m, spec.room_col].mean())); p.append(float(df.loc[m, spec.power_col].mean()))
    return dict(grid=list(dfs), T=t, P=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lbnl", choices=["lbnl", "limassol"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--quick", action="store_true")
    a, _ = ap.parse_known_args()      # 忽略 train_cnn.py 專用參數 (--L, --mono)
    if a.quick:
        a.seeds, a.epochs = 1, 15
    out = Path(a.out or f"out/{a.dataset}"); fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    ens, sc, spec, L = T.load_ensemble(out / "models", "CNN_MLP")
    dfs = spec.load(a.data)
    print(f"[{spec.title}] 載入 CNN 集成: {len(ens)} 個模型, L={L}")
    multi = len(dfs) > 1
    res, plt = {}, setup()

    # A. 留出 (只有多組資料才做)
    hold = None
    if multi:
        res["holdout"], hold = exp_holdout(dfs, spec, ens, sc, L, a.seeds, a.epochs)
        ho = res["holdout"]
        hz = np.arange(1, spec.H + 1) * spec.dt_min / 60
        hm = np.array([x["rollout"] for x in ho["holdout"]])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(hz, ho["seen"]["rollout"], color=COLORS["all"], marker="o", ms=3, label="全部訓練（看過 14/18）")
        ax.plot(hz, hm.mean(0), color=COLORS["holdout"], marker="o", ms=3, label="只用 12/16/20 訓練（沒看過）")
        ax.fill_between(hz, hm.mean(0) - hm.std(0), hm.mean(0) + hm.std(0), color=COLORS["holdout"], alpha=.2)
        ax.plot(hz, ho["persistence"]["rollout"], color=COLORS["persistence"], ls="--", label="persistence")
        ax.set(xlabel="預測時間長度 (h)", ylabel="室溫 RMSE (°C)", title="留出 setpoint 測試（14°C、18°C 檔）")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "fig5_holdout.png", dpi=140); plt.close(fig)

    # B. 控制量響應掃描
    print("== 控制量響應掃描")
    base = dfs[16] if multi else list(dfs.values())[0]
    sw = dict(all=action_sweep(ens, sc, spec, base, L))
    if hold:
        sw["holdout"] = action_sweep(hold[0], hold[1], spec, base, L)
    if multi:
        sw["truth"] = truth_by_group(dfs, spec)
    res["sweep"] = sw
    grid = np.array(spec.sweep_grid)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, key in enumerate(["T", "P"]):
        for name, col, lab in [("all", COLORS["all"], "CNN 集成"), ("holdout", COLORS["holdout"], "留出 14/18 的模型")]:
            if name not in sw:
                continue
            mu, sd = np.array(sw[name][key]), np.array(sw[name][f"{key}_band"])
            ax[i].plot(grid, mu, color=col, lw=1.8, marker="o", ms=3, label=lab)
            ax[i].fill_between(grid, mu - sd, mu + sd, color=col, alpha=.15)
        ax[i].plot([], [], color="#888", lw=6, alpha=.3, label="陰影 = 不同起點（天氣）的變異")
        if "truth" in sw:
            ax[i].scatter(sw["truth"]["grid"], sw["truth"][key], color=COLORS["actual"], s=50, zorder=5,
                          edgecolor="k", label="EnergyPlus 真值（測試月平均）")
        ax[i].set(xlabel=spec.act_label, ylabel="平均室溫 (°C)" if key == "T" else "平均 HVAC 功率 (kW)",
                  title="控制量 → 室溫" if key == "T" else "控制量 → 功率（DE 面對的地形）")
        ax[i].legend(fontsize=8)
    dur = spec.steps_to_h(spec.sweep_H - spec.sweep_burn)
    fig.suptitle(f"固定控制量 rollout {spec.steps_to_h(spec.sweep_H):g} h，取後 {dur:g} h 平均", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "fig6_sweep.png", dpi=140); plt.close(fig)

    # C. 視窗長度消融
    res["ablation"] = {}
    for Lx in (12, 24, 48):
        tr, va, te = hd.build_splits(dfs, spec, Lx)
        scx = hd.Scaler().fit(tr)
        runs = []
        for s in range(a.seeds):
            print(f"== 消融 L={Lx} seed {s}")
            m, h = T.fit("CNN_MLP", tr, va, scx, spec, Lx, seed=s, epochs=a.epochs, verbose=False)
            met = T.one_step_metrics(T.predict(m, scx, te), te)
            met["roll_end"] = float(T.rollout_rmse_curve(m, scx, spec, dfs, Lx)[-1])
            met["params"] = float(sum(p.numel() for p in m.parameters()))
            met["seconds"] = h["seconds"]
            runs.append(met)
        res["ablation"][Lx] = {k: dict(mean=float(np.mean([r[k] for r in runs])), std=float(np.std([r[k] for r in runs])))
                               for k in ("T_rmse", "P_rmse", "roll_end", "params", "seconds")}
    ab = res["ablation"]; Ls = list(ab)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, (k, lab) in enumerate([("T_rmse", "單步室溫 RMSE (°C)"), ("P_rmse", "單步功率 RMSE (kW)"),
                                  ("roll_end", f"{spec.steps_to_h(spec.H):g} h rollout RMSE (°C)")]):
        ax[i].bar([f"{x}\n({spec.steps_to_h(x):g} h)" for x in Ls], [ab[x][k]["mean"] for x in Ls],
                  yerr=[ab[x][k]["std"] for x in Ls], color=COLORS["CNN_MLP"], alpha=.8, capsize=4)
        ax[i].set(xlabel="歷史視窗 L（步）", title=lab)
        ax[i].set_ylim(min(ab[x][k]["mean"] for x in Ls) * 0.8, None)
    fig.tight_layout(); fig.savefig(fig_dir / "fig7_ablation.png", dpi=140); plt.close(fig)

    # D. 不確定度: 測試月的一段長 rollout
    df_u = list(dfs.values())[-1]
    st = T.rollout_starts(df_u, spec, L, spec.long_H)
    t0 = st[len(st) // 3]
    r = T.rollout(ens, sc, spec, df_u, [t0], spec.long_H, L)
    u = dict(T=r["T"][0], T_std=r["T_std"][0], T_true=r["T_true"][0], P=r["P"][0], P_true=r["P_true"][0],
             coverage_2sigma=float(np.mean(np.abs(r["T"][0] - r["T_true"][0]) <= 2 * r["T_std"][0])),
             rmse=T.rmse(r["T"][0], r["T_true"][0]), start=str(df_u.timestamp.iloc[t0 + 1]))
    res["uncertainty"] = u
    hh = np.arange(1, spec.long_H + 1) * spec.dt_min / 60
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(hh, u["T_true"], color=COLORS["actual"], lw=1.2, label="實際")
    ax[0].plot(hh, u["T"], color=COLORS["CNN_MLP"], lw=1.2, label=f"CNN 集成 rollout（RMSE {u['rmse']:.2f} °C）")
    ax[0].fill_between(hh, u["T"] - 2 * u["T_std"], u["T"] + 2 * u["T_std"], color=COLORS["CNN_MLP"], alpha=.25,
                       label=f"±2σ（覆蓋率 {u['coverage_2sigma']:.0%}）")
    ax[0].set(ylabel="室溫 (°C)", title=f"長時間自我回饋 rollout（起點 {u['start'][:16]}，只給初始 {spec.steps_to_h(L):g} h 真值）")
    ax[1].plot(hh, u["P_true"], color=COLORS["actual"], lw=1.2, label="實際")
    ax[1].plot(hh, u["P"], color=COLORS["CNN_MLP"], lw=1.2, label="CNN 集成")
    ax[1].set(ylabel="HVAC 功率 (kW)", xlabel="時間 (h)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig8_uncertainty.png", dpi=140); plt.close(fig)

    # E. 物理約束消融
    if spec.mono_lambda > 0:
        tr, va, te = hd.build_splits(dfs, spec, L)
        free = []
        for s in range(a.seeds):
            print(f"== 無約束對照 seed {s}")
            m, _ = T.fit("CNN_MLP", tr, va, sc, spec, L, seed=s, epochs=a.epochs, mono=0.0, verbose=False)
            free.append(m)
        ce = dict(constrained=T.one_step_metrics(T.predict(ens, sc, te), te),
                  free=T.one_step_metrics(T.predict(free, sc, te), te))
        for name, ms in [("constrained", ens), ("free", free)]:
            ce[name]["rollout"] = T.rollout_rmse_curve(ms, sc, spec, dfs, L)
            ce[name].update(T.mono_violation(ms, sc, spec, te))
        ce["sweep_free"] = action_sweep(free, sc, spec, base, L)
        res["constraint"] = ce
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        for name, col, lab, swd in [("constrained", COLORS["all"], f"有物理約束 (λ={spec.mono_lambda:g})", sw["all"]),
                                    ("free", COLORS["holdout"], "無約束", ce["sweep_free"])]:
            ax[0].plot(grid, swd["T"], color=col, marker="o", ms=3, lw=1.8, label=lab)
            ax[1].plot(grid, swd["P"], color=col, marker="o", ms=3, lw=1.8, label=lab)
            hz = np.arange(1, spec.H + 1) * spec.dt_min
            ax[2].plot(hz, ce[name]["rollout"], color=col, lw=1.8,
                       label=f"{lab}（違反率 {ce[name].get('T_violation', 0):.0%}）")
        ax[0].set(xlabel=spec.act_label, ylabel="平均室溫 (°C)", title="控制量 → 室溫（應單調）")
        ax[1].set(xlabel=spec.act_label, ylabel="平均功率 (kW)", title="控制量 → 功率")
        ax[2].set(xlabel="預測時間長度（分鐘）", ylabel="室溫 RMSE (°C)", title="多步 rollout 誤差的代價")
        for x in ax:
            x.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / "fig9_constraint.png", dpi=140); plt.close(fig)

    T.save_json(res, out / "results" / "eval.json")
    print("\n=== 進階評估摘要 ===")
    if "constraint" in res:
        c = res["constraint"]
        for name in ("constrained", "free"):
            print(f"{name:11s} T {c[name]['T_rmse']:.4f}  roll {c[name]['rollout'][-1]:.3f}  "
                  f"違反率 T {c[name].get('T_violation', 0):.1%} P {c[name].get('P_violation', 0):.1%}")
    if multi:
        ho = res["holdout"]
        print(f"留出 setpoint  T {np.mean([x['T_rmse'] for x in ho['holdout']]):.3f} (看過 {ho['seen']['T_rmse']:.3f})")
    print("掃描 室溫:", [round(x, 2) for x in sw["all"]["T"]])
    print("掃描 功率:", [round(x, 1) for x in sw["all"]["P"]])
    for x in Ls:
        print(f"L={x:2d}  T {ab[x]['T_rmse']['mean']:.4f}  P {ab[x]['P_rmse']['mean']:.2f}  roll {ab[x]['roll_end']['mean']:.3f}")
    print(f"長 rollout RMSE {u['rmse']:.3f}，±2σ 覆蓋率 {u['coverage_2sigma']:.0%}")


if __name__ == "__main__":
    main()
