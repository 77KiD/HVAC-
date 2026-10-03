"""
hvac.controllers — 控制器 (負責人: 模糊控制組)

所有控制器共用介面:  action = ctrl(error, d_error, action_prev)
    error   = T_room - T_target   (°C, 正 = 太熱)
    d_error = error - error_prev
回傳下一步的控制量, closed_loop 會依 spec.act_range clip。
    LBNL     : 閥門開度 0–1      → 太熱 (E>0) 應「開大」閥門
    Limassol : 冷水 setpoint °C  → 太熱 (E>0) 應「調低」setpoint
"""
import numpy as np


class FixedController:
    """Baseline: 固定控制量"""
    def __init__(self, value):
        self.value = value

    def __call__(self, error, d_error, action_prev):
        return self.value


class FuzzyController:
    """
    TODO(模糊控制組): 參考謝旻晃論文 3.3 節
      - 輸入 E, ΔE 各 5 個語意項 (NM NS ZE PS PM), 三角歸屬函數
      - 輸出 Δaction 5 個語意項 (DM DS ZE IS IM), 重心法解模糊
      - action = action_prev + Δaction × scaling factor
      - params 是 DE 要最佳化的向量 (例: 歸屬函數寬度、scaling factor)
    """
    def __init__(self, params=None):
        self.params = np.asarray(params) if params is not None else None

    def __call__(self, error, d_error, action_prev):
        raise NotImplementedError("模糊控制組實作")
