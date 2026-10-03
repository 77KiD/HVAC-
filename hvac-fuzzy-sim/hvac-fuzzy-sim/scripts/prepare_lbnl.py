"""
prepare_lbnl.py — Day 1: 檢查 LBNL SDAHU 欄位 + 前處理

    python scripts/prepare_lbnl.py                    ← 自動搜尋 (建議)
    python scripts/prepare_lbnl.py --src 某路徑        ← 手動指定 zip 或 AHU_annual.csv
    (選用) 檢查故障檔是否重複:  --check-dup zip1 zip2

做的事:
  1. 讀無故障檔 AHU_annual.csv (可直接從 zip 讀, 不用先解壓)
  2. Day 1 檢查清單: 室溫 / 控制量 / 功率或能耗 欄位、常數欄、單位、缺值、時間間隔
  3. 前處理 → data/lbnl_sdahu_5min.csv (5 分鐘、營業時段、公制、估算功率)
  4. 輸出 out/lbnl/results/data_check.json 與 out/lbnl/figures/fig0_data_overview.png
"""
import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hvac.datasets import preprocess_lbnl, LBNL_FILE, COP
from hvac.train import save_json
from hvac.plotting import setup

REQUIRED = {
    "室溫": [f"ZONE_TEMP_{i}" for i in range(1, 6)],
    "控制量": ["CHWC_VLV_DM"],
    "功率或能耗": ["SF_WAT", "RF_WAT"],
}


SKIP_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__", "AppData", "anaconda3", "site-packages"}


def _zip_has_ahu(path):
    """zip 內是否有 AHU_annual.csv (包含 zip 裡再包 zip 的情況)"""
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if any(n.endswith("AHU_annual.csv") for n in names):
                return True
            return any(n.lower().endswith(".zip") and "sdahu" in n.lower() for n in names)
    except (zipfile.BadZipFile, OSError):
        return False


def find_source(max_depth=5):
    """依序在專案、上層資料夾、下載/桌面/文件/OneDrive 找 AHU_annual.csv 或含它的 zip"""
    here = Path.cwd()
    home = Path.home()
    roots = [here / "data", here, here.parent, here.parent.parent,
             home / "Downloads", home / "下載", home / "Desktop", home / "桌面",
             home / "Documents", home / "文件", home / "OneDrive"]
    seen, hits = set(), []
    for root in roots:
        if not root.exists() or root.resolve() in seen:
            continue
        seen.add(root.resolve())
        base_depth = len(root.resolve().parts)
        for dirpath, dirnames, filenames in os.walk(root):
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                fp = Path(dirpath) / f
                if f == "AHU_annual.csv":
                    hits.append((0, fp))
                elif f.lower().endswith(".zip") and "sdahu" in f.lower() and _zip_has_ahu(fp):
                    hits.append((1, fp))
        if hits:
            break
    hits = sorted(set(hits))
    return hits[0][1] if hits else None


def read_raw(src):
    src = Path(src)
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if n.endswith("AHU_annual.csv")]
            if names:
                print(f"從 zip 讀取 {names[0]} …")
                with z.open(names[0]) as f:
                    return pd.read_csv(f)
            # zip 裡再包 zip (例如 *_all_3.zip 內含 SDAHU-1.zip)
            for inner in [n for n in z.namelist() if n.lower().endswith(".zip")]:
                with zipfile.ZipFile(z.open(inner)) as zi:
                    names = [n for n in zi.namelist() if n.endswith("AHU_annual.csv")]
                    if names:
                        print(f"從 {inner} 讀取 {names[0]} …")
                        with zi.open(names[0]) as f:
                            return pd.read_csv(f)
            raise FileNotFoundError(f"{src} 裡沒有 AHU_annual.csv (無故障檔在 SDAHU-1)")
    if not src.exists():
        raise FileNotFoundError(f"找不到 {src}")
    print(f"讀取 {src} …")
    return pd.read_csv(src)


def check_dup(zips):
    """故障檔內容是否重複 (逐檔 MD5)"""
    digests = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if not n.endswith(".csv"):
                    continue
                h = hashlib.md5()
                with z.open(n) as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                digests[Path(n).name] = h.hexdigest()
                print(f"  md5 {Path(n).name}")
    groups = {}
    for name, d in digests.items():
        groups.setdefault(d, []).append(name)
    return [sorted(v) for v in groups.values() if len(v) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="AHU_annual.csv 或含它的 zip；不給就自動搜尋")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out/lbnl")
    ap.add_argument("--check-dup", nargs="*", default=[], help="要檢查重複的故障 zip 檔")
    a = ap.parse_args()

    src = a.src
    if src is None:
        print("自動搜尋 AHU_annual.csv 或 SDAHU zip（專案、下載、桌面、文件、OneDrive）…")
        src = find_source()
        if src is None:
            raise SystemExit(
                "\n找不到 LBNL 資料。請把 LBNL_FDD_Data_Sets_SDAHU 的 zip（或解壓出的 AHU_annual.csv）\n"
                "複製到專案的 data 資料夾後再執行一次：  python scripts\\prepare_lbnl.py")
        print(f"找到: {src}")
    raw = read_raw(src)
    raw["Datetime"] = pd.to_datetime(raw["Datetime"])
    num = raw.drop(columns="Datetime")

    # ─── Day 1 檢查清單 ───
    chk = {"rows": len(raw), "start": str(raw.Datetime.min()), "end": str(raw.Datetime.max()),
           "interval_min": float(raw.Datetime.diff().dt.total_seconds().median() / 60),
           "missing": int(num.isna().sum().sum()), "dup_timestamps": int(raw.Datetime.duplicated().sum())}
    chk["required"] = {k: {c: c in raw.columns for c in cols} for k, cols in REQUIRED.items()}
    chk["constant_cols"] = [c for c in num.columns if num[c].std() < 1e-6]
    occ = raw[raw.SYS_CTL == 1]
    chk["valve"] = dict(mean=float(occ.CHWC_VLV_DM.mean()), std=float(occ.CHWC_VLV_DM.std()),
                        frac_open=float((occ.CHWC_VLV_DM > 0.01).mean()))
    chk["units"] = {
        "SA_SP_mean": float(raw.SA_SP.mean()),
        "SA_SP_note": "平均約 400 → 實際為 Pa (1.6 inH₂O ≈ 398 Pa)，文件標示 inH₂O 有誤",
        "SA_CFM_median": float(occ.SA_CFM.median()),
        "SA_CFM_note": "若為 CFM 則冷卻負載達數千 kW，不合理；以 L/min 解讀時約 90 kW，合理",
    }
    chk["occupied"] = dict(
        fraction=float((raw.SYS_CTL == 1).mean()),
        weekdays=sorted(int(x) for x in occ.Datetime.dt.dayofweek.unique()),
        hours=[int(occ.Datetime.dt.hour.min()), int(occ.Datetime.dt.hour.max())],
        note="0=週一。資料的營業日/時與文件 (週一–六 6–22 時) 相差約 1 天 1 小時，推測為模擬年份與日曆不同",
    )
    if a.check_dup:
        print("檢查故障檔是否重複 (每個 zip 約 1–2 分鐘) …")
        chk["duplicate_fault_files"] = check_dup(a.check_dup)

    # ─── 前處理 ───
    df = preprocess_lbnl(raw)
    Path(a.data).mkdir(exist_ok=True)
    dst = Path(a.data) / LBNL_FILE
    df.to_csv(dst, index=False)
    chk["processed"] = dict(file=str(dst), rows=len(df), interval_min=5,
                            room_temp_c=[float(df.room_temp_c.min()), float(df.room_temp_c.max())],
                            power_kw_mean=float(df.power_kw.mean()), cool_kw_mean=float(df.cool_kw.mean()),
                            fan_kw_mean=float(df.fan_kw.mean()), cop_assumed=COP)
    ok = all(all(v.values()) for v in chk["required"].values())
    chk["decision"] = ("採用 LBNL SDAHU：室溫、控制量 (CHWC_VLV_DM) 齊全；功率只有風扇實測，"
                       "冷卻用電以冷卻負載 ÷ COP 估算") if ok else "缺必要欄位 → 改用備案資料"
    out = Path(a.out)
    save_json(chk, out / "results" / "data_check.json")

    # ─── 圖: 9 月某一週 ───
    plt = setup()
    wk = df[(df.timestamp >= "2018-09-10") & (df.timestamp < "2018-09-17")]
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for i in range(1, 6):
        ax[0].plot(wk.timestamp, wk[f"zone{i}_c"], lw=.6, alpha=.5)
    ax[0].plot(wk.timestamp, wk.room_temp_c, color="#EEEEEE", lw=1.4, label="5 區平均 (模型目標)")
    ax0b = ax[0].twinx(); ax0b.plot(wk.timestamp, wk.oa_temp_c, color="#FFB74D", lw=1, label="室外溫度")
    ax0b.set_ylabel("室外 (°C)", color="#FFB74D"); ax0b.grid(False)
    ax[0].set(ylabel="室溫 (°C)", title="LBNL SDAHU 前處理後資料（2018/9/10–9/16，只保留營業時段）")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(wk.timestamp, wk.valve, color="#4FC3F7", lw=1, label="冷卻閥門 CHWC_VLV_DM (控制量)")
    ax[1].plot(wk.timestamp, wk.oa_damper, color="#81C784", lw=1, alpha=.7, label="外氣風門")
    ax[1].set(ylabel="開度"); ax[1].legend(fontsize=8)
    ax[2].plot(wk.timestamp, wk.cool_kw / COP, color="#E57373", lw=1, label=f"冷卻用電估算 (負載÷COP {COP:g})")
    ax[2].plot(wk.timestamp, wk.fan_kw, color="#BA68C8", lw=1, label="風扇實測")
    ax[2].set(ylabel="kW"); ax[2].legend(fontsize=8)
    fig.tight_layout(); (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "fig0_data_overview.png", dpi=140); plt.close(fig)

    # ─── 終端機清單 ───
    print("\n========== Day 1 資料檢查 ==========")
    print(f"期間 {chk['start']} ~ {chk['end']}，{chk['rows']:,} 筆，每 {chk['interval_min']:.0f} 分鐘，缺值 {chk['missing']}")
    for k, cols in chk["required"].items():
        print(f"  {'✔' if all(cols.values()) else '✘'} {k}: {', '.join(cols)}")
    print(f"  閥門 (營業時段): 平均 {chk['valve']['mean']:.2f}、標準差 {chk['valve']['std']:.2f}、"
          f"開啟比例 {chk['valve']['frac_open']:.0%}")
    print(f"  常數欄 (無用): {', '.join(chk['constant_cols'])}")
    print(f"  單位: SA_SP 平均 {chk['units']['SA_SP_mean']:.0f} → Pa；SA_CFM 以 L/min 解讀")
    if "duplicate_fault_files" in chk:
        for g in chk["duplicate_fault_files"]:
            print(f"  ⚠ 內容完全相同: {' = '.join(g)}")
    p = chk["processed"]
    print(f"\n前處理 → {p['file']}：{p['rows']:,} 筆 (5 分鐘)，室溫 {p['room_temp_c'][0]:.1f}–{p['room_temp_c'][1]:.1f} °C")
    print(f"  平均功率 {p['power_kw_mean']:.1f} kW (風扇 {p['fan_kw_mean']:.2f} + 冷卻負載 {p['cool_kw_mean']:.1f} ÷ COP {COP:g})")
    print(f"\n決定: {chk['decision']}")


if __name__ == "__main__":
    main()
