from typing import List, Tuple, Optional

import torch.utils.data
from torch import nn

from generator.conv_2d_weight_modulate import Conv2dWeightModulate


class StyleBlock(nn.Module):
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        # TODO: EqualizedLinearは後で作成
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # なぜ0.2?
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        """
        :param x: 特徴量のmap. shapeは[batch_size, in_features, height, width]
        :param w: weight. shapeは[batch_size, d_latent]
        :param noise: tensor. shapeは[batch_size, 1, height, width]
        :return:
        """
        # style vector
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])
