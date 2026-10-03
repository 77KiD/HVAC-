"""hvac.models — 動態模型 (負責人: CNN 組)"""
import torch
import torch.nn as nn


class CNN_MLP(nn.Module):
    """CNN 分支吃過去 L 小時序列; MLP 分支吃下一刻 action + 天氣; 合併後輸出 [ΔT, P]"""
    def __init__(self, c_seq, d_now, L, hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(c_seq, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
        )
        self.cnn_fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * L, hidden), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(d_now, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 32, hidden), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, x_seq, x_now):                         # x_seq: [B, L, C]
        z = self.cnn_fc(self.cnn(x_seq.transpose(1, 2)))     # Conv1d 要 [B, C, L]
        return self.head(torch.cat([z, self.mlp(x_now)], dim=1))


class MLPOnly(nn.Module):
    """對照組: 同樣的輸入, 但直接攤平, 沒有卷積"""
    def __init__(self, c_seq, d_now, L, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c_seq * L + d_now, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 2))

    def forward(self, x_seq, x_now):
        return self.net(torch.cat([x_seq.flatten(1), x_now], dim=1))


ARCHS = {"CNN_MLP": CNN_MLP, "MLPOnly": MLPOnly}
