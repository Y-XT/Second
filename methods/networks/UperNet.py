import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from mmseg.models.decode_heads import UPerHead

class UPerDispHead(nn.Module):
    """
    UPerHead 融合 + disparity 输出。
    默认 compat='monodepth2'：返回与 Monodepth2 完全一致的金字塔：
      ("disp",0): H×W, ("disp",1): H/2×W/2, ("disp",2): H/4×W/4, ("disp",3): H/8×W/8
    选 compat='upernet'：返回 UPerNet 传统尺度：
      ("disp",0): C2 尺度(≈H/4), ("disp",1): H/8, ("disp",2): H/16, ("disp",3): H/32
    """
    def __init__(
        self,
        in_channels,                 # 例: [96, 192, 384, 768]
        channels=256,                # FPN中间通道
        pool_scales=(1, 2, 3, 6),
        align_corners=False,
        dropout_ratio=0.1,
        compat: str = "monodepth2",  # "monodepth2" | "upernet"
        out_scales=(0,1,2,3),        # 需要哪些尺度键
        upsample_mode: str = "bilinear",
        downsample_mode: str = "area"
    ):
        super().__init__()
        assert len(in_channels) == 4, "需要4个尺度特征 [C2,C3,C4,C5]"
        assert compat in ("monodepth2", "upernet")
        self.align_corners = align_corners
        self.compat = compat
        self.out_scales = tuple(sorted(out_scales))
        self.upsample_mode = upsample_mode
        self.downsample_mode = downsample_mode

        # 复用 UPerHead（num_classes=1 即回归1通道）
        self.decode = UPerHead(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=align_corners,
            pool_scales=pool_scales
        )

        # 轻量平滑
        self.post = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)

    @staticmethod
    def _sigmoid_disp(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _build_pyramid_monodepth2(
        self,
        disp_base: torch.Tensor,
        image_size: Optional[Tuple[int, int]],
        c2_hw: Tuple[int, int],
    ):
        """
        从任意尺度的单通道 disp 出发，构建 Monodepth2 金字塔。
        若 image_size 未给，按常规 ConvNeXt 假设从 C2(h4,w4) 反推 H=4*h4, W=4*w4。
        """
        h4, w4 = c2_hw
        if image_size is None:
            H, W = 4 * h4, 4 * w4
        else:
            H, W = image_size
        # 断言可被 8 整除（Monodepth2 常见假设；如不满足，请在数据管线里pad/crop）
        assert H % 8 == 0 and W % 8 == 0, "H,W 建议能被 8 整除以匹配 Monodepth2 的 4 级金字塔"

        # 先上采到 full-res
        disp0 = F.interpolate(disp_base, size=(H, W), mode=self.upsample_mode, align_corners=self.align_corners)

        # 逐级 area 下采样
        disp1 = F.interpolate(disp0, size=(H // 2, W // 2), mode=self.downsample_mode)
        disp2 = F.interpolate(disp0, size=(H // 4, W // 4), mode=self.downsample_mode)
        disp3 = F.interpolate(disp0, size=(H // 8, W // 8), mode=self.downsample_mode)

        out = {}
        if 0 in self.out_scales: out[("disp", 0)] = disp0
        if 1 in self.out_scales: out[("disp", 1)] = disp1
        if 2 in self.out_scales: out[("disp", 2)] = disp2
        if 3 in self.out_scales: out[("disp", 3)] = disp3
        return out

    def _build_pyramid_upernet(self, disp_1_4: torch.Tensor):
        """
        UPerNet 传统：主输出在 C2 尺度(≈H/4)，再向下采样得到 1/8、1/16、1/32。
        """
        base = disp_1_4
        h4, w4 = base.shape[2:]
        sizes = {
            0: (h4,        w4),        # 1/4
            1: (h4 // 2,   w4 // 2),   # 1/8
            2: (h4 // 4,   w4 // 4),   # 1/16
            3: (h4 // 8,   w4 // 8),   # 1/32
        }
        out = {}
        for s in (0,1,2,3):
            if s in self.out_scales:
                out[("disp", s)] = base if s == 0 else F.interpolate(base, size=sizes[s], mode=self.downsample_mode)
        return out

    def forward(self, feats, image_size: Optional[Tuple[int, int]] = None):
        """
        feats: [C2,C3,C4,C5]
        image_size: (H,W)。compat='monodepth2' 下：
          - 若为 None：按 C2 尺度反推 H=4*h4,W=4*w4；
          - 若不为 None：强制用该 H,W。
        """
        # 主输出（通常 C2 尺度）
        x = self.decode(feats)          # [B,1,h4,w4]
        x = self.post(x)
        disp_1_4 = self._sigmoid_disp(x)

        if self.compat == "monodepth2":
            # feats[0] 即 C2，获取其空间尺寸
            c2 = feats[0]
            h4, w4 = c2.shape[2], c2.shape[3]
            return self._build_pyramid_monodepth2(disp_1_4, image_size, (h4, w4))
        else:
            return self._build_pyramid_upernet(disp_1_4)


if __name__ == "__main__":
    # 演示：ConvNeXt 常见 C2~C5 尺度（H/4 开始）
    B, H, W = 2, 480, 640
    feats = [
        torch.randn(B,  96, H//4,  W//4),   # C2
        torch.randn(B, 192, H//8,  W//8),   # C3
        torch.randn(B, 384, H//16, W//16),  # C4
        torch.randn(B, 768, H//32, W//32),  # C5
    ]

    # 1) 默认：Monodepth2 兼容输出（无需传 image_size，自动反推）
    head_md2 = UPerDispHead(in_channels=[96,192,384,768], compat="monodepth2").eval()
    with torch.no_grad():
        out_md2 = head_md2(feats)  # 将返回 H×W, H/2, H/4, H/8 四级
    for k, v in out_md2.items():
        print("MD2", k, v.shape)

    # 2) 可选：标准 UPerNet 结构的尺度契约（1/4,1/8,1/16,1/32）
    head_up = UPerDispHead(in_channels=[96,192,384,768], compat="upernet").eval()
    with torch.no_grad():
        out_up = head_up(feats)
    for k, v in out_up.items():
        print("UPN", k, v.shape)
