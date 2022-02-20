import torch
from torch import nn


class MiniBatchStdDev(nn.Module):
    """
    各特徴量についてミニバッチ全体の標準偏差を計算する。
    その後全ての標準偏差の平均を取り、それを1つの特徴量として特徴量mapに追加する
    """

    def __init__(self, group_size: int = 4):
        """
        :param group_size: 標準偏差を計算するためのサンプル数
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        """
        :param x: 特徴量のmap
        """
        assert x.shape[0] % self.group_size == 0
        # サンプルを group_size のグループに分割し、各特徴の標準偏差を計算したいので、特徴量マップを一次元に平坦化
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # 平均標準偏差
        std = std.mean().view(1, 1, 1, 1)
        # 標準偏差を展開し、特徴量のmapに追加
        b, _, h, w = x.shape
        std = std.expend(b, -1, h, w)
        # 標準偏差を特徴量マップに追加（連結）する
        return torch.cat([x, std], dim=1)
