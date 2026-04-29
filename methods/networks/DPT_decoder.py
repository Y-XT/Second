# methods/networks/DPT_decoder.py 里的类：用这一版覆盖
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

from mmseg.models.decode_heads import DPTHead


class MMSegDPTMonoDepth2Head(nn.Module):
    """
    输入: feats = [C1(×4), C2(×8), C3(×16), C4(×32)]
    输出: 与 Monodepth2 对齐的 {("disp", s): tensor}，s∈{0,1,2,3}；值域 Sigmoid。
    """
    def __init__(
        self,
        in_channels: Tuple[int, int, int, int] = (96, 192, 384, 768),
        channels: int = 256,
        out_channel: int = 1,
        align_corners: bool = False,
    ):
        super().__init__()
        self.align_corners = align_corners

        # 仅传入“各版本通用且必要”的最小参数：
        # - in_channels: 四级特征通道
        # - in_index:    对应特征索引 (0,1,2,3)
        # - channels:    DPT 中间通道
        # - num_classes: 输出通道（这里=1，视差）
        # - align_corners: 与你的插值设置保持一致
        # - input_transform: 'multiple_select'，声明输入为多特征
        self.head = DPTHead(
            in_channels=list(in_channels),
            in_index=(0, 1, 2, 3),
            channels=channels,
            num_classes=out_channel,
            align_corners=align_corners,
            input_transform='multiple_select',
        )

        self.activation = nn.Sigmoid()

    def forward(self, feats: List[torch.Tensor]) -> Dict[Tuple[str, int], torch.Tensor]:
        assert isinstance(feats, (list, tuple)) and len(feats) == 4, \
            f"期望 4 个尺度特征 [C1×4, C2×8, C3×16, C4×32]，实际 len={len(feats)}"
        C1, C2, C3, C4 = feats
        H, W = C1.shape[-2] * 4, C1.shape[-1] * 4  # C1 为 1/4 尺度

        # DPTHead 期望 tuple 顺序与 in_index 对应
        logit_1_4 = self.head(inputs=(C1, C2, C3, C4))   # [B, 1, H/4, W/4]
        disp_1_4  = self.activation(logit_1_4)

        out: Dict[Tuple[str, int], torch.Tensor] = {}
        out[("disp", 2)] = disp_1_4
        out[("disp", 3)] = F.interpolate(disp_1_4, scale_factor=0.5,
                                         mode="bilinear", align_corners=self.align_corners)
        out[("disp", 1)] = F.interpolate(disp_1_4, scale_factor=2.0,
                                         mode="bilinear", align_corners=self.align_corners)
        out[("disp", 0)] = F.interpolate(disp_1_4, size=(H, W),
                                         mode="bilinear", align_corners=self.align_corners)
        return out
