import torch.nn.functional as F
import torch.utils.data
from torch import nn

from parameter.equalized_weight import EqualizedWeight


class Conv2dWeightModulate(nn.Module):
    """
    このレイヤーはconvolution weightをスタイルベクトルでスケーリングし、それを正規化することで復調
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        """
        :param in_features: 入力特徴量mapの特徴量数
        :param out_features: 出力特徴量mapの特徴量数
        :param kernel_size: convolution kernelのsize
        :param demodulate: 重みを標準偏差で正規化するかどうかのフラグ
        :param eps: 正規化する時に使うepsilon
        """
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.eps = eps
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        :param x: 入力特徴量のmap. shapeは[batch_size, in_features, height, width]
        :param s: style based scaling tensor. shapeは[batch_size, in_features]
        :return: shapeは[batch_size, out_features, height, width]
        """
        batch_size, _, height, width = x.shape
        # reshape
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        # [batch_size, out_features, in_features, kernel_size, kernel_size]のshapeの値を返す
        weights = weights * s
        # 正規化
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv
        x = x.reshape(1, -1, height, width)
        # reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(batch_size * self.out_features, *ws)
        # グループ化されたコンボリューションを使用して、サンプル単位のカーネルでコンボリューションを効率的に計算する。
        # つまり、バッチ内の各サンプルに対して異なるカーネル（重み）を持つ。
        x = F.conv2d(x, weights, padding=self.padding, groups=batch_size)
        return x.reshape(-1, self.out_features, height, width)
