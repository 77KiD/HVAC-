from pathlib import Path
import pytest

DATA = Path(__file__).resolve().parents[1] / "data"


def _load(name, probe):
    if not (DATA / probe).exists():
        pytest.skip(f"缺少 data/{probe}")
    from hvac.datasets import get_spec
    spec = get_spec(name)
    return spec, spec.load(DATA)


@pytest.fixture(scope="session", params=["lbnl", "limassol"])
def ds(request):
    """兩份資料集都跑同一組測試 (缺檔就跳過那一份)"""
    probe = "lbnl_sdahu_5min.csv" if request.param == "lbnl" else "baseline_12C_results.csv"
    return _load(request.param, probe)
