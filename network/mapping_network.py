import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class MappingNetwork(nn.Module):

    def __init__(self, features: int, n_layers: int):
        """
        :param features:  zとwの特徴量の数
        :param n_layers:  mapping networkのレイヤー数
        """
        super().__init__()
        layers = []
        for i in range(n_layers):
            # TODO: EqualizedLinear後で説明がある
            layers.append(EqualizedLinear(features, features))
            # LeakyReLU:
            # 入力値が0より下の場合には入力値をnegative_slope値分だけ小さくする
            # 0以上の場合はそのまま
            # 理由は複雑らしい
            # https://atmarkit.itmedia.co.jp/ait/articles/2005/13/news009.html
            # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=1)
        return self.net(z)


