"""一鍵跑完 CNN 部分: 訓練 → 進階評估 → HTML 報告

    python scripts/run_cnn_pipeline.py                        # LBNL 完整
    python scripts/run_cnn_pipeline.py --quick                # 只確認流程能跑
    python scripts/run_cnn_pipeline.py --dataset limassol     # 備案資料
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
args = sys.argv[1:]
report_args = []
for i, x in enumerate(args):
    if x in ("--dataset", "--out") and i + 1 < len(args):
        report_args += [x, args[i + 1]]
for script, extra in [("train_cnn.py", args), ("evaluate_cnn.py", args), ("make_report.py", report_args)]:
    print(f"\n######## {script} {' '.join(extra)}", flush=True)
    r = subprocess.run([sys.executable, str(here / script), *extra])
    if r.returncode != 0:
        sys.exit(f"{script} 失敗 (exit {r.returncode})")
print("\n完成。打開 out/<dataset>/cnn_report.html 看報告。")
