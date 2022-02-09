from typing import List, Tuple, Optional

import torch.utils.data
from torch import nn

from generator.ToRGB import ToRGB
from generator.style_block import StyleBlock


class GeneratorBlock(nn.Module):
    """
    2つのスタイルブロック（スタイル変調による3×3コンボリューション）とRGB出力から構成されている
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        :param d_latent: wの次元数
        :param in_features: 入力特徴量mapの特徴量数
        :param out_features: 出力特徴量mapの特徴量数
        """
        super().__init__()
        # 最初のスタイルブロックは、feature map sizeをout_featuresに変更します。
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        :param x: shapeが [batch_size, in_features, height, width] の入力特徴マップ
        :param w: shapeが [batch_size, d_latent] のw
        :param noise: shapeが [batch_size, 1, height, width] のノイズ
        """
        # 最初のノイズテンソルを用いた最初のスタイルブロック。出力は [batch_size, out_features, height, width] のshape
        x = self.style_block1(x, w, noise[0])
        # 2番目ののノイズテンソルを用いた2番目のスタイルブロック。出力は [batch_size, out_features, height, width] のshape
        x = self.style_block2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, rgb
