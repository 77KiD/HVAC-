"""hvac.plotting — 統一的深色圖表風格"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"CNN_MLP": "#4FC3F7", "MLPOnly": "#FFB74D", "persistence": "#9E9E9E",
          "actual": "#EEEEEE", "holdout": "#E57373", "all": "#81C784"}


def setup():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#161b22", "axes.facecolor": "#0d1117",
        "savefig.facecolor": "#161b22", "axes.edgecolor": "#30363d",
        "grid.color": "#30363d", "axes.grid": True, "grid.alpha": 0.6,
        "font.size": 10, "axes.titlesize": 11, "legend.framealpha": 0.3,
        "font.family": "sans-serif",
        # Windows / macOS / Linux 的中文字型, 依序嘗試
        "font.sans-serif": ["Microsoft JhengHei", "PingFang TC", "Heiti TC",
                            "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    return plt
