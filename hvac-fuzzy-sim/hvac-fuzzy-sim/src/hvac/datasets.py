"""
hvac.datasets — 資料集規格與讀取。全專案透過 Spec 取得欄位名稱, 換資料集只要換 spec。

    from hvac.datasets import get_spec
    spec = get_spec("lbnl")          # 或 "limassol"
    dfs  = spec.load("data")         # dict{group: DataFrame}

每個 DataFrame 都有: timestamp, seg (連續段編號), group, h_sin, h_cos, 以及 spec 指定的欄位。
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Spec:
    name: str
    title: str
    seq_cols: list          # CNN 歷史視窗通道 (只能放「控制量 + 室溫 + 外生變數」, 不能放受控制影響的中間量)
    now_cols: list          # MLP 分支: 下一步的控制量 + 外生變數
    action_col: str
    room_col: str
    power_col: str
    act_range: tuple        # 控制量範圍, 模擬器會 clip
    act_label: str
    dt_min: int             # 時間步長 (分鐘)
    L: int                  # 預設歷史視窗步數
    H: int                  # 多步 rollout 步數
    train_months: tuple
    val_months: tuple
    test_months: tuple
    comfort_target: float
    comfort_band: tuple
    group_label: str = ""
    sweep_grid: tuple = ()
    sweep_H: int = 0
    sweep_burn: int = 0
    long_H: int = 0
    # 物理單調性約束: 持續改變控制量時, 室溫/功率應有的變化方向 (+1 上升, -1 下降, 0 不約束)
    mono_T: int = 0
    mono_P: int = 0
    mono_lambda: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def room_idx(self): return self.seq_cols.index(self.room_col)
    @property
    def act_idx(self): return self.seq_cols.index(self.action_col)
    @property
    def act_now_idx(self): return self.now_cols.index(self.action_col)
    def steps_to_h(self, k): return k * self.dt_min / 60

    def load(self, data_dir="data"):
        return LOADERS[self.name](Path(data_dir))


def _add_time(df, gap_min):
    h = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    df["h_sin"] = np.sin(2 * np.pi * h / 24)
    df["h_cos"] = np.cos(2 * np.pi * h / 24)
    gap = df.timestamp.diff().dt.total_seconds().div(60).fillna(0)
    df["seg"] = (gap > gap_min).cumsum()
    return df


# ───────────────────────── Limassol (EnergyPlus 飯店, 每小時) ─────────────────────────
LIMASSOL_SETPOINTS = [12, 14, 16, 18, 20]


def load_limassol(data_dir):
    out = {}
    for sp in LIMASSOL_SETPOINTS:
        f = data_dir / f"baseline_{sp}C_results.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} 不存在, 先執行 python scripts/download_data.py")
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        df = _add_time(df, gap_min=90)
        df["group"] = sp
        out[sp] = df
    return out


# ───────────────────────── LBNL SDAHU (單風管空調箱, 每分鐘 → 5 分鐘) ─────────────────────────
LBNL_FILE = "lbnl_sdahu_5min.csv"
F2C = lambda f: (f - 32) * 5 / 9
COP = 3.0                 # 冰水主機 COP 假設, 用來把冷卻負載換成電功率
RHO_CP = 1.2 * 1.006      # 空氣密度 (kg/m³) × 比熱 (kJ/kg·K)


def preprocess_lbnl(raw):
    """原始 1 分鐘 CSV → 5 分鐘、只保留營業時段、公制單位"""
    raw = raw.copy()
    raw["Datetime"] = pd.to_datetime(raw["Datetime"])
    d = raw.set_index("Datetime").resample("5min").mean()
    d = d[d["SYS_CTL"] == 1]                               # 5 分鐘內全部為營業時段
    zones = [f"ZONE_TEMP_{i}" for i in range(1, 6)]
    out = pd.DataFrame(index=d.index)
    out["room_temp_c"] = F2C(d[zones].mean(axis=1))        # 5 區平均室溫
    for i, z in enumerate(zones, 1):
        out[f"zone{i}_c"] = F2C(d[z])
    out["oa_temp_c"] = F2C(d["OA_TEMP"])
    out["ma_temp_c"] = F2C(d["MA_TEMP"])
    out["sa_temp_c"] = F2C(d["SA_TEMP"])
    out["valve"] = d["CHWC_VLV_DM"].clip(0, 1)             # 控制量: 冷卻盤管閥門指令
    out["oa_damper"] = d["OA_DMPR_DM"].clip(0, 1)          # 外氣風門 (由 economizer 控制, 視為外生)
    out["fan_kw"] = ((d["SF_WAT"] + d["RF_WAT"]) / 1000).clip(lower=0)
    vdot = d["SA_CFM"] / 60000                              # 欄位實際單位為 L/min → m³/s
    out["cool_kw"] = (RHO_CP * vdot * (out["ma_temp_c"] - out["sa_temp_c"])).clip(lower=0)
    out["power_kw"] = out["fan_kw"] + out["cool_kw"] / COP  # 估算 HVAC 電功率
    out = out.dropna().reset_index().rename(columns={"Datetime": "timestamp"})
    return out


def load_lbnl(data_dir):
    f = data_dir / LBNL_FILE
    if not f.exists():
        raise FileNotFoundError(f"{f} 不存在, 先執行 python scripts/prepare_lbnl.py --src <AHU_annual.csv 或 zip>")
    df = pd.read_csv(f, parse_dates=["timestamp"])
    df = _add_time(df, gap_min=5)
    df["group"] = "fault_free"
    return {"fault_free": df}


LOADERS = {"limassol": load_limassol, "lbnl": load_lbnl}

SPECS = {
    "limassol": Spec(
        name="limassol", title="Limassol Hotel（EnergyPlus，每小時）",
        seq_cols=["room_temp_c", "outdoor_temp_c", "solar_w_m2", "chiller_setpoint_c", "h_sin", "h_cos"],
        now_cols=["chiller_setpoint_c", "outdoor_temp_c", "solar_w_m2", "h_sin", "h_cos"],
        action_col="chiller_setpoint_c", room_col="room_temp_c", power_col="hvac_total_kw",
        act_range=(12.0, 20.0), act_label="冷水 setpoint (°C)", dt_min=60, L=24, H=24,
        train_months=(5, 6, 7, 8), val_months=(9,), test_months=(10,),
        comfort_target=23.0, comfort_band=(22.5, 25.0), group_label="setpoint 檔",
        sweep_grid=tuple(np.arange(12, 20.01, 0.5)), sweep_H=168, sweep_burn=24, long_H=168,
        mono_T=+1, mono_P=0, mono_lambda=0.0,
        notes=["每個 CSV 的 setpoint 全程固定，模型沒看過 setpoint 切換的暫態。",
               "室溫多數時間被恆溫器鎖在 23 °C，單步預測難以超越 persistence。"]),
    "lbnl": Spec(
        name="lbnl", title="LBNL SDAHU 單風管空調箱（Chicago，5 分鐘）",
        seq_cols=["room_temp_c", "oa_temp_c", "valve", "oa_damper", "h_sin", "h_cos"],
        now_cols=["valve", "oa_temp_c", "oa_damper", "h_sin", "h_cos"],
        action_col="valve", room_col="room_temp_c", power_col="power_kw",
        act_range=(0.0, 1.0), act_label="冷卻閥門開度 (0–1)", dt_min=5, L=24, H=24,
        train_months=(1, 2, 3, 4, 5, 6, 7, 10, 11, 12), val_months=(8,), test_months=(9,),
        comfort_target=22.5, comfort_band=(21.1, 23.9), group_label="情境",
        sweep_grid=tuple(np.round(np.arange(0, 1.001, 0.1), 2)), sweep_H=36, sweep_burn=12, long_H=144,
        mono_T=-1, mono_P=+1, mono_lambda=1.0,
        notes=["功率為估算值：風扇實測功率 + 冷卻負載 ÷ COP（假設 COP = 3）；冷卻負載 = ρ·cp·風量·(混風溫 − 送風溫)。",
               "SA_CFM 的實際單位推斷為 L/min（文件標示 CFM），SA_SP 為 Pa（文件標示 inH₂O）。",
               "外氣風門由 economizer 控制，在模擬器中視為已知外生變數。",
               "只使用營業時段；非營業時段風扇關閉，室溫會漂到 30 °C 以上。",
               "閉迴路資料的混淆：原 PI 控制器在熱的時候開大閥門，未加約束的模型會學成「閥門開大 → 室溫較高」；"
               "因此訓練時加入物理單調性約束（閥門開大 → 室溫下降、功率上升）。",
               "閥門直接控制的是送風溫度（資料中 90% 以上時間維持在 12.88 °C），室溫另由 VAV 末端箱調節（資料未提供），"
               "閥門對室溫的影響是間接的。"]),
}


def get_spec(name):
    if name not in SPECS:
        raise KeyError(f"未知資料集 {name}，可用: {list(SPECS)}")
    return SPECS[name]
