import numpy as np
from hvac import data as hd


def test_sorted_unique_and_columns(ds):
    spec, dfs = ds
    for df in dfs.values():
        assert df.timestamp.is_monotonic_increasing
        assert not df.timestamp.duplicated().any()
        for c in set(spec.seq_cols + spec.now_cols + [spec.power_col]):
            assert c in df and df[c].notna().all(), c


def test_action_in_range(ds):
    spec, dfs = ds
    lo, hi = spec.act_range
    for df in dfs.values():
        assert df[spec.action_col].between(lo, hi).all()


def test_windows_and_split(ds):
    spec, dfs = ds
    tr, va, te = hd.build_splits(dfs, spec)
    assert tr["X_seq"].shape[1:] == (spec.L, len(spec.seq_cols))
    assert set(np.unique(tr["month"])) <= set(spec.train_months)
    assert set(np.unique(te["month"])) <= set(spec.test_months)
    assert np.allclose(tr["X_seq"][:, -1, spec.room_idx], tr["T_now"])
    assert len(te["y"]) > 1000


def test_scaler_roundtrip(ds):
    spec, dfs = ds
    tr, _, _ = hd.build_splits(dfs, spec)
    sc = hd.Scaler().fit(tr)
    y = tr["y"][:10]
    assert np.allclose(sc.y_inv((y - sc.y_m) / sc.y_s), y, atol=1e-4)


def test_lbnl_preprocess_units():
    """LBNL 前處理: 華氏→攝氏, 估算功率 ≥ 0"""
    import pandas as pd
    from hvac.datasets import preprocess_lbnl
    t = pd.date_range("2018-07-02 08:00", periods=10, freq="1min")
    raw = pd.DataFrame({"Datetime": t, "SYS_CTL": 1.0, "OA_TEMP": 95.0, "MA_TEMP": 80.0,
                        "SA_TEMP": 55.0, "CHWC_VLV_DM": 0.5, "OA_DMPR_DM": 0.1,
                        "SF_WAT": 1000.0, "RF_WAT": 500.0, "SA_CFM": 600000.0,
                        **{f"ZONE_TEMP_{i}": 75.0 for i in range(1, 6)}})
    out = preprocess_lbnl(raw)
    assert len(out) == 2
    assert abs(out.room_temp_c.iloc[0] - 23.889) < 1e-3
    assert out.power_kw.ge(0).all() and out.cool_kw.iloc[0] > 0
