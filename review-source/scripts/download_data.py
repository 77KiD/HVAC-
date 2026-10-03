"""下載 Limassol Hotel HVAC 資料集 (CC0) 到 data/"""
import sys
import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/NeuraEnergy/"
        "limassol-hotel-hvac-energyplus-dataset/main/data/")
SETPOINTS = [12, 14, 16, 18, 20]


def main(out_dir="data"):
    out = Path(out_dir); out.mkdir(exist_ok=True)
    for sp in SETPOINTS:
        name = f"baseline_{sp}C_results.csv"
        dst = out / name
        if dst.exists() and dst.stat().st_size > 100_000:
            print(f"skip  {name} (已存在)")
            continue
        print(f"get   {name}")
        urllib.request.urlretrieve(BASE + name, dst)
    print("完成 →", out.resolve())


if __name__ == "__main__":
    main(*sys.argv[1:])
