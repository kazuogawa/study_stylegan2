import math

from torch import nn

from layer.equalized_conv_2d import EqualizedConv2d


class DiscriminatorBlock(nn.Module):
    """
    識別器ブロックは、2つの3×3畳み込みと残差接続で構成
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # residual: 残留
        # ダウンサンプリングと残差接続のための1×1畳み込み層
        # TODO: DownSampleは後で
        self.residual = nn.Sequential(DownSample(), EqualizedConv2d(in_features, out_features, kernel_size=1))
        # 3 x 3の畳み込み層
        self.block = nn.Sequential(EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   EqualizedConv2d(out_features, out_features, kernel_size=3, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # TODO: DownSampleは後で
        self.down_Sample = DownSample()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_Sample(x)
        x = (x + residual) * self.scale
        return x
