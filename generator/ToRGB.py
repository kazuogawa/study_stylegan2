import torch.utils.data
from torch import nn

from generator.conv_2d_weight_modulate import Conv2dWeightModulate
from layer.equalized_linear import EqualizedLinear


class ToRGB(nn.Module):
    """
    1×1畳み込みにより、特徴マップからRGB画像を生成
    """

    def __init__(self, d_latent: int, features: int):
        super().__init__()
        self.features = features
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # なぜ3なのか
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        :param x: 特徴量のmap. shapeは[batch_size, in_features, height, width]
        :param w: shapeは[batch_size, d_latent]
        :return:
        """
        # style vector
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])
