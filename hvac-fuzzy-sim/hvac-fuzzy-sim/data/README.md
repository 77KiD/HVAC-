# LBNL 資料放置說明

原始 LBNL SDAHU 資料較大，不隨 GitHub 儲存庫提供。每位組員請使用自己已有的資料，將整個 `LBNL_FDD_Dataset_SDAHU` 資料夾放到本目錄。

請確認無故障資料檔的相對路徑為：

```text
hvac-fuzzy-sim/hvac-fuzzy-sim/data/LBNL_FDD_Dataset_SDAHU/AHU_annual.csv
```

若解壓後多一層資料夾，請調整位置，確保 `AHU_annual.csv` 直接位於上述資料夾。前處理只使用這份無故障資料；故障案例 CSV 可留在同一資料夾。

從儲存庫根目錄進入實際程式目錄，再執行：

```powershell
cd hvac-fuzzy-sim/hvac-fuzzy-sim
python scripts/prepare_lbnl.py --src data/LBNL_FDD_Dataset_SDAHU/AHU_annual.csv
```

前處理會產生或覆寫 `data/lbnl_sdahu_5min.csv`，檢查結果位於 `out/lbnl/results/data_check.json`。

每位組員首次使用都必須執行上述前處理，再進行訓練。`lbnl_sdahu_5min.csv` 為本機產生的檔案，不隨 GitHub 提供；若本機已有舊版，也請重新前處理覆寫。原始 CSV 更新或 `preprocess_lbnl` 修改後，必須再次前處理。

成功時請確認：

1. `data/lbnl_sdahu_5min.csv` 已產生且非空。
2. `out/lbnl/results/data_check.json` 已更新，包含檢查與處理筆數。
3. 再執行 `python scripts/run_cnn_pipeline.py --dataset lbnl --quick --out out/lbnl_recheck` 確認流程。

原始資料夾已由儲存庫根目錄的 `.gitignore` 排除。請勿強制加入 Git。`scripts/download_data.py` 是 Limassol 備案下載工具，不會下載 LBNL。
