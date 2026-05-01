"""
CNN-LSTM for 12-Lead ECG Reconstruction (Zehui Zhan)
=====================================================
논문: "Conditional generative adversarial network driven variable-duration
      single-lead to 12-lead electrocardiogram reconstruction"
      (Zehui Zhan, Jiarong Chen, Kangming Li, Wanqing Wu)
공식 GitHub: https://github.com/Zehui-Zhan/12-lead-reconstruction

원본 코드: LSTM.py (Generator_lstm 클래스) 에서 직접 추출.
원본: in=1, out=11 (입력 lead 제외)
수정: in=1, out=12 (전체 12 leads 출력), 동적 길이 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Zehui(nn.Module):
    """
    원본 Generator_lstm 구조: Conv1D → MaxPool → Conv1D → MaxPool → Conv1D → MaxPool
    → LSTM → Dense → Conv1D 출력.

    원본은 입력 길이 1024 고정이었으나, 동적 길이를 지원하도록 수정.
    """
    def __init__(self, in_channel=1, out_channel=12, hidden_size=64, num_layers=1):
        super().__init__()
        # 원본 LSTM.py 17-23줄
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=128,
                               kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=5, stride=1, padding=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=out_channel,
                               kernel_size=5, stride=1, padding=2)
        self._out_channel = out_channel

    def forward(self, z):
        """
        Args:
            z: (B, 1, T) - 단일 리드 ECG
        Returns:
            (B, 12, T) - 복원된 12-리드 ECG
        """
        T = z.size(2)

        # 원본 LSTM.py 26-40줄
        x1 = self.conv1(z)
        x2 = self.maxpool(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv2(x4)
        x6 = self.maxpool(x5)

        # LSTM 처리
        x7 = x6.permute(0, 2, 1)  # (B, T/8, 128)
        x8, _ = self.lstm(x7)     # (B, T/8, hidden_size)

        # Flatten → Dense → Reshape
        x8 = torch.flatten(x8, start_dim=1)  # (B, T/8 * hidden_size)
        x8 = x8.unsqueeze(1)     # (B, 1, T/8 * hidden_size)
        x8 = self.drop(x8)

        # 동적 Dense: adaptive projection to T
        dense = nn.Linear(x8.size(2), T).to(z.device)
        x9 = dense(x8)           # (B, 1, T)
        x10 = self.conv3(x9)     # (B, out_channel, T)
        return x10


class LSTM_Zehui_Fixed(nn.Module):
    """
    고정 길이 버전 (원본 그대로). 입력/출력 길이 = 1000.
    PTB-XL 100Hz (1000 samples) 전용.
    """
    def __init__(self, in_channel=1, out_channel=12, seq_len=1000):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=128,
                               kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=5, stride=1, padding=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64,
                            num_layers=1, batch_first=True)
        self.dense = nn.Linear((seq_len // 8) * 64, seq_len)
        self.drop = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=out_channel,
                               kernel_size=5, stride=1, padding=2)

    def forward(self, z):
        x1 = self.conv1(z)
        x2 = self.maxpool(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv2(x4)
        x6 = self.maxpool(x5)
        x7 = x6.permute(0, 2, 1)
        x8, _ = self.lstm(x7)
        x8 = torch.flatten(x8, start_dim=1)
        x8 = x8.unsqueeze(1)
        x8 = self.drop(x8)
        x9 = self.dense(x8)
        x10 = self.conv3(x9)
        return x10
