"""CNN 部分的單元測試 (需要 torch; 沒裝會自動跳過)"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hvac import data as hd
from hvac import train as T
from hvac.models import ARCHS
from hvac.simulator import HVACSimulator


@pytest.mark.parametrize("arch", list(ARCHS))
@pytest.mark.parametrize("L", [12, 24, 48])
def test_model_shapes(arch, L):
    m = ARCHS[arch](6, 5, L)
    assert m(torch.randn(7, L, 6), torch.randn(7, 5)).shape == (7, 2)


@pytest.fixture(scope="module")
def tiny(ds, tmp_path_factory):
    """2 個 epoch 的迷你模型, 存成 2 個 seed 的集成"""
    spec, dfs = ds
    L = 12
    tr, va, te = hd.build_splits(dfs, spec, L=L)
    sc = hd.Scaler().fit(tr)
    d = tmp_path_factory.mktemp(f"models_{spec.name}")
    models = []
    for s in range(2):
        m, h = T.fit("CNN_MLP", tr, va, sc, spec, L, seed=s, epochs=2, verbose=False)
        T.save_ckpt(d / f"cnn_mlp_s{s}.pt", "CNN_MLP", spec, L, m, sc)
        models.append(m)
    return dict(dir=d, models=models, sc=sc, spec=spec, dfs=dfs, L=L, te=te, hist=h)


def test_history_recorded(tiny):
    h = tiny["hist"]
    assert len(h["train"]) == len(h["val"]) == 2 and "best_epoch" in h


def test_ckpt_roundtrip(tiny):
    m2, sc2, spec2, ck = T.load_ckpt(tiny["dir"] / "cnn_mlp_s0.pt")
    assert spec2.name == tiny["spec"].name and ck["L"] == tiny["L"]
    p1 = T.predict(tiny["models"][0], tiny["sc"], tiny["te"])
    p2 = T.predict(m2, sc2, tiny["te"])
    assert np.allclose(p1, p2, atol=1e-5)


def test_simulator_ensemble_step(tiny):
    spec = tiny["spec"]
    sim = HVACSimulator.load(tiny["dir"])
    assert len(sim.models) == 2 and sim.spec.name == spec.name
    df = next(iter(tiny["dfs"].values()))
    seq = df[spec.seq_cols].to_numpy(np.float32)
    window, nxt = seq[100:100 + sim.L], seq[100 + sim.L]
    lo, hi = spec.act_range
    T_next, P_next = sim.step(window, (lo + hi) / 2, nxt)
    (Tm, Pm), (Ts, Ps) = sim.step_dist(window, (lo + hi) / 2, nxt)
    assert abs(T_next - Tm) < 1e-6 and Ts >= 0
    assert 5 < T_next < 45 and np.isfinite(P_next)
    assert sim.step(window, hi + 99, nxt) == sim.step(window, hi, nxt)   # clip


def test_rollout_shapes_and_override(tiny):
    spec = tiny["spec"]
    df = next(iter(tiny["dfs"].values()))
    st = T.rollout_starts(df, spec, tiny["L"], 6)[:5]
    r = T.rollout(tiny["models"], tiny["sc"], spec, df, st, 6, tiny["L"])
    assert r["T"].shape == r["T_true"].shape == (5, 6)
    lo, hi = spec.act_range
    r_lo = T.rollout(tiny["models"], tiny["sc"], spec, df, st, 6, tiny["L"], act_override=lo)
    r_hi = T.rollout(tiny["models"], tiny["sc"], spec, df, st, 6, tiny["L"], act_override=hi)
    assert not np.allclose(r_lo["P"], r_hi["P"])     # 控制量改變應影響功率
