from typing import List, Tuple, Optional

import torch.utils.data
from torch import nn

from generator.ToRGB import ToRGB
from generator.generator_block import GeneratorBlock


class Generator(nn.Module):
    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        """
        :param log_resolution: log2の画像解像度
        :param d_latent: wの次元
        :param n_features: 最高解像度での畳み込みレイヤーの特徴量の数（最終ブロック）
        :param max_features: 任意のジェネレータブロックの最大の特徴量の数
        """
        super().__init__()
        # 各ブロックの特徴量の数を計算。下記のようなもの
        #  [512, 512, 256, 128, 64, 32]
        features: list[int] = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks: int = len(features)
        # torch.Size([1, 512, 4, 4])のランダム定数
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        # 4×4解像度の最初のスタイルブロック
        # TODO: StyleBlockは後で追加
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])
        self.blocks = nn.ModuleList(
            [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        )
        # 2×アップサンプリング層。特徴量の空間はブロック毎にアップサンプリングされる
        # TODO: UpSampleは後で追加
        self.up_sample = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        """
        :param w: 各generator blockに用意した別々のw。レイヤーごとに異なるwを使用し、mix-styleするために使う。
        shapeは[n_blocks, batch_size, d_latent]
        :param input_noise: 入力ノイズ。shapeは[n_blocks, batch_size, d_latent]
        :return:
        """
        batch_size: int = w.shape[1]
        # バッチサイズに合わせて学習した定数を拡張する
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        # first style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # first rgb image
        rgb = self.to_rgb(x, w[0])
        # 残りのブロックを評価
        for i in range(1, self.n_blocks):
            # 特徴量の空間をアップサンプリング
            x = self.up_sample(x)
            # generator blockを評価
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            # RGB画像をアップサンプリングし、ブロックからのrgbに追加する
            rgb = self.up_sample(rgb) + rgb_new
        return rgb
