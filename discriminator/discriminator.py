import torch.utils.data
from torch import nn

from layer.equalized_conv_2d import EqualizedConv2d
from layer.equalized_linear import EqualizedLinear


class Discriminator(nn.Module):
    def __init__(self, log_resolution: int, n_features: int, max_features: int = 512):
        """
        :param log_resolution: log2の画像解像度
        :param n_features: 最高解像度での畳み込み層の特徴数（1ブロック目に当たる解像度のこと）
        :param max_features: generator blockの最大の特徴量数
        """
        super().__init__()
        # RGB 画像を n_features 個の特徴数を持つ特徴マップに変換するレイヤー
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 各ブロックの特徴数を計算。[64, 128, 256, 512, 512, 512]のようになるらしい
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        # discriminator blockの数
        n_blocks = len(features) - 1
        # discriminator block
        # TODO: DiscriminatorBlockは後で実装する
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        # mini-batchの標準偏差
        # TODO: MiniBatchStdDevは後で実装する
        self.std_dev = MiniBatchStdDev()
        # 標準偏差マップ追加後の特徴量数
        final_features: int = features[-1] + 1
        # 最後の畳み込みlayer
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        # 最後のLeakyReLU
        #
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: input_imageのshape. [batch_size, 3, height, width]
        """
        # 画像の正規化を試みる（これは完全にオプションですが、初期のトレーニングを少し早めるらしい）
        x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        # flatten
        x = x.reshape(x.shape[0], -1)
        return self.final(x)
