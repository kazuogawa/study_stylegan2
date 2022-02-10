import torch.nn.functional as F
import torch.utils.data
from torch import nn

from parameter.equalized_weight import EqualizedWeight


class EqualizedLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        # learning-rate equalized weights
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # ただの線形関数
        return F.linear(x, self.weight(), bias=self.bias)
