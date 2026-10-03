# LBNL SDAHU：資料前處理與 1D-CNN+MLP 驗證及交接

查核日期：2026-10-03。版本依據：使用者提供的 hvac-fuzzy-sim.zip，內層 hvac-fuzzy-sim/hvac-fuzzy-sim；結果以 out/lbnl 為準。

本文件核對現有程式與儲存結果，未重新訓練、執行測試或重算原始資料。文獻為本次補查的方法依據，不代表開發時確實引用。範圍為資料前處理、下一刻室溫預測、MLP-only 對照與多步 rollout，不包含模糊控制及 DE。

## 1. 任務與完成狀態

| 項目 | 可查核完成標準 | 現有證據及狀態 |
|---|---|---|
| 決定資料集 | 有區域溫度、控制量、功率欄位，並記錄使用決定 | data_check.json 記錄採用 SDAHU 無故障 AHU_annual.csv |
| 前處理 | 說明單位、取樣、缺值與連續段處理；保存處理資料 | datasets.py 與 lbnl_sdahu_5min.csv 已存在；風量單位尚待確認 |
| 模型與對照 | 相同輸入、資料切分與評估條件；有模型權重 | CNN_MLP 與 MLPOnly 權重及 main.json 已存在 |
| 預測圖與 rollout | 有預測對照圖與逐 horizon 誤差 | fig2_pred_vs_actual.png、fig4_rollout.png 已存在；未在本次重算或檢查圖像 |
| 正式效能驗收 | 事前約定門檻或清楚呈現比較證據 | 尚無指定門檻；可記錄產物完成及單次實驗結果，不能宣告正式性能達標 |

## 2. 資料來源與欄位

LBNL 官方將 Single-duct AHU 列為 EnergyPlus／Modelica 產生的模擬資料，不能稱為現場量測資料。[官方資料頁](https://faultdetection.lbl.gov/data/)

| 原始欄位 | 用途 | 原始單位／處理 |
|---|---|---|
| ZONE_TEMP_1 至 ZONE_TEMP_5 | 五區域平均室溫，主要預測目標 | °F，換算 °C 後平均 |
| CHWC_VLV_DM | 冷卻盤管閥門控制指令 | 0–1；不是閥門位置 CHWC_VLV |
| OA_TEMP | 外氣溫度 | °F → °C |
| OA_DMPR_DM | 外氣風門控制指令 | 0–1；評估時視為已知輸入 |
| SF_WAT、RF_WAT | 送風及回風風扇功率 | W → kW |
| MA_TEMP、SA_TEMP、SA_CFM | 冷卻負載估算 | 溫度 °F；風量單位有疑義 |
| SYS_CTL | occupied mode 篩選 | 保留五分鐘平均值等於 1 的時段 |

欄位依據為 [SDAHU 官方說明表 2，PDF 第 6–7 頁](https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_SDAHU/LBNL_FDD_Data_Sets_SDAHU.pdf)。其中 SA_CFM 官方標示 CFM；程式依數值量級推斷為 L/min。這個推斷尚未獲來源確認，不應寫成官方誤植已被證實。

保存的檢查結果：原始 525,540 筆，2018-01-01 01:00 至 2018-12-31 23:59，標示每分鐘一筆、缺值及重複時間戳為 0；前處理後 55,425 筆。這些數字來自 data_check.json，非本次獨立重算。

前處理依序為：解析時間、每五分鐘平均、occupied 篩選、單位轉換、控制量截限、估算功率、刪除含缺值列。讀取處理檔時將超過五分鐘的缺口分為不同連續段，歷史視窗不得跨段。原始檢查紀錄無重複，不代表程式已具備一般性的重複時間處理；重採樣前也未逐欄實施離群值清理。

## 3. 公式與依據對照

以下公式以程式為準；符號 T 為五區平均室溫，a 為閥門指令，P 為估算功率。

| 公式 | 程式位置 | 來源性質 |
|---|---|---|
| T_C=(T_F−32)×5/9；T_room=(Σ T_zone,i)/5 | datasets.py：F2C、preprocess_lbnl | 單位定義及本專案等權平均；並非特定論文提出 |
| h_sin=sin(2πh/24)，h_cos=cos(2πh/24) | datasets.py：_add_time | 本專案週期時間編碼 |
| z=(x−μ_train)/(σ_train+10⁻⁶) | data.py：Scaler | 標準化定義；只用訓練資料估計統計量 |
| y_t=[T_(t+1)−T_t, P_(t+1)]；T̂_(t+1)=T_t+ΔT̂_(t+1) | data.py：make_windows；simulator.py | 本專案差分目標與重建方式 |
| Conv 輸出：b_o+Σ_c Σ_j W_(o,c,j) x_(c,i+dj−p)；越界補零 | models.py：CNN_MLP | [PyTorch Conv1d 運算依據](https://docs.pytorch.org/docs/main/generated/torch.conv1d.html)；實際為互相關形式 |
| MLP：z=ReLU(Wx+b)，最後線性層輸出兩個數值 | models.py：CNN_MLP、MLPOnly | 標準前饋網路運算，具體架構是本專案選擇；[官方教學](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html?highlight=conv2d) |
| L_data=(1/(2B))Σ_i[(ΔT̂'_i−ΔT'_i)²+0.5(P̂'_i−P'_i)²] | train.py：fit | 標準化輸出加權平方誤差；分母 2B 對應程式對兩個輸出與 batch 一起 mean。權重是專案設定；[MSE 定義](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss) |
| RMSE=√[(1/N)Σ_i(ŷ_i−y_i)²] | train.py：rmse、one_step_metrics | 評估指標定義；ΔT 誤差等於以同一真實 T_t 重建後的單步 T 誤差 |
| R²=1−Σ(y−ŷ)²/Σ(y−ȳ)² | scripts/train_cnn.py：fig3 | 評估指標定義，須說明分母為零時不可用 |
| RMSE(h)=√[(1/N)Σ_i(T̂_(i,h)−T_(i,h))²] | train.py：rollout_rmse_curve | 專案逐 horizon 評估；不是把整條曲線平均後的單一 RMSE |
| persistence：T̂_(t+h)=T_t | data.py、train.py | 最後觀測值基準，不需訓練 |
| P_fan=(SF_WAT+RF_WAT)/1000 | datasets.py | 功率單位轉換 |
| Q_sensible=max(ρc_p V̇(T_ma−T_sa),0)；P_est=P_fan+Q_sensible/COP | datasets.py | 穩態顯熱能量平衡與 COP 定義；只估算顯熱，未計潛熱。ρ=1.2 kg/m³、c_p=1.006 kJ/(kg·K)、COP=3 是專案假設，非資料集提供或已校準係數 |

程式採 V̇=SA_CFM/60000，只有在原值確為 L/min 時才能得到 m³/s；若單位確為 CFM，應乘約 0.00047194745。總功率標籤因此仍屬附帶假設的估算值，不能當作真實整套 HVAC 電功率。

### 單調性正則化的精確形式

模型亦使用單調性懲罰，而非只有 MSE。令 G_j 為同時增加歷史全部閥門指令及下一步指令的方向導數：

G_j=σ_y,j[Σ_ℓ (∂ŷ'_j/∂x'_(ℓ,a))/σ_seq,a +(∂ŷ'_j/∂x'_now,a)/σ_now,a]。

L_mono=mean[ReLU(G_T/σ_y,T)²]+mean[ReLU(−G_P/σ_y,P)²]；L_train=L_data+λL_mono，λ=1。

σ 在此指 Scaler 實際保存的 std+10⁻⁶。該懲罰鼓勵閥門增加時預測溫度下降、功率上升；它是軟性局部限制，不保證全域單調，也不證明因果控制效果。驗證 loss 僅計 L_data，未包含此懲罰，因此訓練與驗證 loss 曲線的定義不同。程式位置：train.py 的 action_effect、mono_penalty、fit。

可引用 [Monteiro et al.（2022）Monotonicity regularization](https://proceedings.mlr.press/v180/monteiro22a.html) 作為方法背景；本專案的歷史與下一步梯度加總、符號及權重為自身實作，不能寫成直接重現該論文。

## 4. 模型與實驗設定

歷史輸入欄位順序：[room_temp_c, oa_temp_c, valve, oa_damper, h_sin, h_cos]；下一刻輸入：[valve, oa_temp_c, oa_damper, h_sin, h_cos]。輸出為 [ΔT, P]。

L=24 個五分鐘觀測值（約兩小時歷史；首末觀測時間差為 115 分鐘），H=24 步（未來 120 分鐘）。CNN 三層通道 32，kernel=3，dilation=1/2/4，padding=1/2/4，之後 flatten→64；下一刻分支為 5→32；串接後 96→64→2，ReLU、Dropout=0.1。MLP-only 使用同一歷史與下一刻輸入，攤平後 149→128→64→2，亦有 Dropout=0.1。兩者不是參數量匹配的對照，只能比較這兩個具體架構。

卷積時間建模可參考 Bai, Kolter & Koltun（2018），[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)。本專案沒有其完整因果卷積及殘差 TCN 架構；應稱為帶 dilation 的 1D-CNN+MLP，不宣稱完整重現 TCN。

Adam 的方法來源為 Kingma & Ba，[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)。程式預設 lr=0.001、weight_decay=10⁻⁵、batch=256，驗證停滯時學習率乘 0.5，early stopping patience=12。

儲存的主實驗：1 seed、最多 15 epochs、λ=1；與 --quick 設定相符，但沒有命令紀錄可證明實際呼叫參數。訓練 39,901、驗證 4,317、測試 3,935 個視窗。訓練月為 1–7、10–12 月，驗證 8 月，測試 9 月。這是月份留出評估，訓練包含測試月份之後的資料，不能稱為僅用過去預測未來的部署模擬。視窗以目標月份分組，沒有額外跨分割邊界的隔離區間。

## 5. 現存結果與可支持的結論

數值取自 out/lbnl/results/main.json。

| 模型 | 單步室溫 RMSE（°C） | 第 24 步／120 分鐘 RMSE（°C） | 估算功率 RMSE（kW） |
|---|---:|---:|---:|
| CNN_MLP | 0.010130 | 0.254751 | 1.088887 |
| MLPOnly | 0.010898 | 0.264073 | 1.015033 |
| persistence | 0.047582 | 0.690866 | 未提供 |

現有單次實驗中 CNN 的室溫誤差較低；功率項則 MLP-only 較低。只有一個 seed，結果中 std=0 僅表示沒有多 seed 差異可估計，不能解讀為穩定性或不確定度為零。

rollout 以預測室溫回填歷史視窗，但未來控制量、外氣溫度與外氣風門採資料紀錄真值。預設從測試月份可用起點抽最多 300 個，不跨連續段；其第一步誤差不必與全測試集單步 RMSE 一致。這是已知未來輸入條件下的遞迴預測，不能直接代表未知天氣下的部署效果。起點月份檢查亦未逐步強制未來每點皆屬測試月，正式重跑宜確認邊界。

既有長 rollout 的成員標準差全為零（單模型），且存在負功率預測。該部分不能作有效信賴區間或直接用於實際能耗判定。

建議驗收採兩層：產物驗收要求資料、權重、圖表、指標與可重現命令齊備；效能驗收由接收者先訂情境與容許誤差，再用隔離測試資料評估。不以已看到的數字反推任意合格門檻。若需強化研究結論，先確認風量單位，再以多 seeds 重跑並報告平均與標準差、補充按時間先後切分的結果。

## 6. 交接與重現

保留壓縮檔中的 src、scripts、tests、pyproject.toml、requirements.txt、原始 AHU_annual.csv、處理 CSV、out/lbnl/models、results、figures 與 cnn_report.html。根目錄 out 及 out/limassol 是其他結果，不應混用。隨附 .venv 不作可攜環境依據；正式交接應記錄 Python 與實際套件版本。

在解壓後專案根目錄，以全新環境執行以下現有命令；本次尚未執行。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/prepare_lbnl.py --src data/LBNL_FDD_Dataset_SDAHU/AHU_annual.csv
python -m pytest
python scripts/run_cnn_pipeline.py --dataset lbnl --quick --out out/lbnl_recheck
```

--quick 只驗證流程及單次小規模結果。正式預設可移除 --quick，改用獨立輸出資料夾避免覆蓋已交付結果；預設主實驗為 5 seeds、最多 80 epochs。

載入介面以 simulator.py 為準，README 的舊介面不可直接使用：

```python
from hvac.simulator import HVACSimulator
sim = HVACSimulator.load("out/lbnl/models")
T_next, P_next = sim.step(window, action, next_row)
```

window 為 [24,6]、未標準化的實際單位，依 sim.spec.seq_cols 排序；next_row 為長度 6 的下一步資料列，提供下一步外生值。action 是閥門指令 0–1。step_dist 的標準差是模型成員分歧，不是已校準的預測信賴區間。

## 7. 引用清單及來源責任

1. Granderson et al.（2023）. A labeled dataset for building HVAC systems operating in faulted and fault-free states. Scientific Data, 10, 342. [DOI:10.1038/s41597-023-02197-w](https://pmc.ncbi.nlm.nih.gov/articles/PMC10235024/)。作為資料集論文引用。
2. LBNL. Single-Duct Air Handling Unit 資料說明，[官方 PDF](https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_SDAHU/LBNL_FDD_Data_Sets_SDAHU.pdf)。作為欄位及單位依據。
3. Bai, S., Kolter, J. Z., & Koltun, V.（2018）. [卷積序列建模論文](https://arxiv.org/abs/1803.01271)。只作方法背景，非完整架構來源。
4. Kingma, D. P., & Ba, J.（2014 預印本）. [Adam](https://arxiv.org/abs/1412.6980)。作為最佳化方法來源。
5. Monteiro et al.（2022）. [Monotonicity regularization: Improved penalties and novel applications to disentangled representation learning and robust classification](https://proceedings.mlr.press/v180/monteiro22a.html)。作為正則化背景。
6. PyTorch 官方文件及教學：上表提供運算連結；本次線上文件版本不代表模型訓練所用版本。

尚待來源確認：SA_CFM 真實單位、風量轉換的原始依據、固定 COP=3 與空氣性質係數的專案適用性。本次未找到開發時採用的原始設計論文，因此不捏造論文頁碼、公式編號或宣稱上述特定架構出自某一篇論文。
