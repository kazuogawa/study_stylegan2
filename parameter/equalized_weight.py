import math
from typing import List, Tuple, Optional

import numpy as np
import torch.utils.data
from torch import nn


class EqualizedWeight(nn.Module):
    """
    Progressive GANの論文で紹介された均等化学習率に基づくもの。
    重みを N(0,c) で初期化する代わりに、重みを N(0,1) に初期化し、使用時に c を乗算している。
    w_i = c \hat{w}_i
    格納されたパラメータに対する勾配\hat{w} はcを乗じるが、
    Adamのようなオプティマイザは二乗勾配の実行平均で正規化するので、これは影響がない。
    オプティマイザの更新は \hat{w} は学習率λに比例するが、effective weightのwはcλに比例して更新される。
    学習率を均等にしないと、有効重みがλだけ比例して更新されることになる。
    つまり、これらの重みパラメータに対して、事実上、学習率をcでスケーリングしているのである。
    """

    def __init__(self, shape: List[int]):
        super().__init__()
        # 初期化定数らしい。なぜこの式になるのか不明
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c
