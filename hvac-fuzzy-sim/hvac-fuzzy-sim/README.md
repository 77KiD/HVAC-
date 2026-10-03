# HVAC 模擬器專題 — CNN 動態模型 + 模糊控制 + DE

目前採用 LBNL SDAHU 無故障資料訓練 1D-CNN+MLP，並以 MLP-only 與 persistence 對照。原始資料由組員自行放入 `data/LBNL_FDD_Dataset_SDAHU`，詳見 [資料放置說明](data/README.md)。Limassol 是備案。

後續規劃以 1D-CNN 當「受控體模擬器」，
再以模糊控制器決定冷水 setpoint，最後用 DE 最佳化模糊參數，比較溫度誤差與能耗。

## 1. 環境建置（每位組員做一次）

需要 Python 3.10–3.12。建議把專案放在沒有中文、不在 OneDrive 的路徑，例如 `C:\dev\hvac-fuzzy-sim`。

Windows（cmd）：

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts\prepare_lbnl.py --src data/LBNL_FDD_Dataset_SDAHU/AHU_annual.csv
pytest
```

macOS / Linux：把第二行換成 `source .venv/bin/activate`，路徑分隔用 `/`。

`pytest` 全部 passed 就完成。有 NVIDIA GPU 可先依 https://pytorch.org 裝 CUDA 版 torch；模型很小，CPU 也足夠。

## 2. 專案結構

```
hvac-fuzzy-sim/
├── src/hvac/
│   ├── data.py          前處理、時間切分、Scaler             (CNN)
│   ├── models.py        CNN_MLP、MLPOnly                      (CNN)
│   ├── train.py         訓練、預測、存讀檔、多步 rollout      (CNN)
│   ├── simulator.py     HVACSimulator（集成）← 組間合約        (CNN)
│   ├── plotting.py      統一深色圖表風格
│   ├── controllers.py   FixedController、FuzzyController      (模糊組)
│   └── closed_loop.py   閉環迴圈 + metrics                    (Day 3)
├── scripts/
│   ├── download_data.py      下載資料
│   ├── check_data.py         資料筆數與 persistence 對照
│   ├── run_cnn_pipeline.py   ★ 一鍵跑完 CNN 全部（下面三支）
│   ├── train_cnn.py          主實驗：CNN vs MLP vs persistence，多 seed
│   ├── evaluate_cnn.py       留出 setpoint、setpoint 掃描、視窗消融、不確定度
│   ├── make_report.py        整合成 out/cnn_report.html
│   └── run_closed_loop.py    固定 setpoint baseline 範例
├── tests/               pytest
├── data/                CSV（不進 git）
└── out/                 models/  figures/  results/  cnn_report.html（不進 git）
```

## 3. CNN 部分

```bat
python scripts\run_cnn_pipeline.py --quick
python scripts\run_cnn_pipeline.py
```

第一行約 1 分鐘，只確認流程能跑；第二行是正式版（CPU 約 15–30 分鐘）。

| 輸出 | 內容 |
|---|---|
| `out/models/cnn_mlp_s0~4.pt` | 5 個 seed，集成後就是交付的模擬器 |
| `out/figures/fig1_loss.png` | 訓練/驗證 loss 曲線 |
| `out/figures/fig2_pred_vs_actual.png` | 10 月單步預測 vs 實際 |
| `out/figures/fig3_scatter.png` | 預測 vs 實際散佈圖 + R² |
| `out/figures/fig4_rollout.png` | 多步 rollout 誤差（含 persistence） |
| `out/figures/fig5_holdout.png` | 留出 14/18 °C 的插值測試 |
| `out/figures/fig6_sweep.png` | setpoint → 室溫 / 功率響應曲線 vs EnergyPlus 真值 |
| `out/figures/fig7_ablation.png` | 視窗長度 L = 12 / 24 / 48 |
| `out/figures/fig8_uncertainty.png` | 一週 rollout ± 2σ |
| `out/results/main.json`、`eval.json` | 所有數字 |
| `out/cnn_report.html` | 深色互動報告（圖已內嵌，可直接分享） |

## 4. 組間介面（改之前先在群組講）

```python
from hvac.simulator import HVACSimulator
sim = HVACSimulator.load("out/models")            # 集成；也可給單一 .pt
T_next, P_next = sim.step(window, setpoint, outdoor_next, solar_next, hour_next)
(T, P), (T_std, P_std) = sim.step_dist(...)       # 加上集成不確定度

setpoint = controller(error, d_error, sp_prev)    # error = T_room - 23，正 = 太熱
```

- `window` 的欄位順序見 `hvac.data.SEQ_COLS`
- setpoint 一律 clip 在 12–20 °C（訓練資料範圍）
- 模型還沒訓練好之前，可用 `tests/test_closed_loop.py` 的 `DummySim` 開發控制器

## 5. Git 協作

- 每人開自己的分支：`feat/cnn`、`feat/fuzzy`、`feat/de`
- 合併到 `main` 前先跑 `pytest`
- `data/`、`out/`、`*.pt` 不上傳；模型權重用雲端硬碟分享（整個 `out/models` 資料夾）

## 6. 已知資料限制（寫進報告）

- CSV 未依時間排序、有重複時間戳（`hvac.data.load_all` 已處理）
- 實際涵蓋 2025/5/1–10/31，約每小時一筆
- 每檔 setpoint 固定 → 模型沒學過 setpoint 切換的暫態，閉環結果需保守解讀
- 室溫多數時間被恆溫器維持在 23 °C，單步預測很難贏 persistence，評估以多步 rollout 與功率為主
