import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import ConvBlock, Conv3x3

class DepthDecoderDINO(nn.Module):
    def __init__(
        self,
        num_ch_enc,
        scales=None,                  # 推荐传 [0,1,2,3]
        num_output_channels=1,
        use_skips=True,
        dec_base_ch=16,
        upsample_mode="nearest",
        align_corners=False,
        output_full_res=True          # N==4 时，额外输出 s=0 的全分辨率
    ):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.output_full_res = output_full_res

        self.num_ch_enc = np.array(list(map(int, num_ch_enc)), dtype=np.int64)
        self.N = int(len(self.num_ch_enc))
        assert self.N >= 2, "num_ch_enc 至少两层"

        # 解码侧通道：与层数对齐
        self.num_ch_dec = np.array([dec_base_ch * (2 ** i) for i in range(self.N)], dtype=np.int64)

        # 仅导出 4 个尺度（全/1/2/1/4/1/8）
        self.scales = [0, 1, 2, 3] if scales is None else list(scales)

        # --- 构建模块 ---
        convs = OrderedDict()
        for i in range(self.N - 1, -1, -1):
            # upconv_0
            num_ch_in = int(self.num_ch_enc[-1]) if i == (self.N - 1) else int(self.num_ch_dec[i + 1])
            num_ch_out = int(self.num_ch_dec[i])
            convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = int(self.num_ch_dec[i])
            if self.use_skips and i > 0:
                num_ch_in += int(self.num_ch_enc[i - 1])
            convs[("upconv", i, 1)] = ConvBlock(num_ch_in, int(self.num_ch_dec[i]))

        # ★ 注意：dispconv 按 “i 层” 全量构建，避免与 s_key 耦合
        for i in range(self.N):
            convs[("dispconv", i)] = Conv3x3(int(self.num_ch_dec[i]), self.num_output_channels)

        self.convs = convs
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def _upsample_to(self, x, hw):
        if self.upsample_mode == "bilinear":
            return F.interpolate(x, size=hw, mode="bilinear", align_corners=self.align_corners)
        return F.interpolate(x, size=hw, mode=self.upsample_mode)

    def forward(self, input_features):
        assert isinstance(input_features, (list, tuple)) and len(input_features) == self.N
        for f, c in zip(input_features, self.num_ch_enc):
            assert f.shape[1] == int(c)

        outputs = {}
        x = input_features[-1]

        # 推断原图大小：N>=5 → 顶层 /2；N==4 → 顶层 /4
        top_stride = 2 if self.N >= 5 else 4
        full_hw = (input_features[0].shape[-2] * top_stride,
                   input_features[0].shape[-1] * top_stride)

        # N==4 时将 i→s 的键整体右移 1；N>=5 时 i 与 s 一致
        scale_offset = 0 if self.N >= 5 else 1

        for i in range(self.N - 1, -1, -1):
            x = self.convs[("upconv", i, 0)](x)

            if self.use_skips and i > 0:
                tgt = input_features[i - 1]
                x = self._upsample_to(x, tgt.shape[-2:])
                x = torch.cat([x, tgt], dim=1)
            else:
                # 最顶层（i==0）无 skip：再放大 2×（N==4 时得到 /2；N>=5 时得到 /1）
                h, w = x.shape[-2:]
                x = self._upsample_to(x, (h * 2, w * 2))

            x = self.convs[("upconv", i, 1)](x)

            # ---- 主输出：写入 s_key = i + offset（N==4: i=3→s=4 被丢弃）----
            s_key = i + scale_offset
            if s_key in self.scales:
                disp_i = self.sigmoid(self.convs[("dispconv", i)](x))
                # N>=5：s_key==0 就是全分辨率；N==4：s_key==1 是 1/2
                outputs[("disp", s_key)] = disp_i

            # ---- 额外补充：N==4 时再提供 s=0（全分辨率） ----
            if self.N == 4 and self.output_full_res and (0 in self.scales) and i == 0:
                # 先得到 /2 的 disp（上面 disp_i），再升到全分辨率
                disp_full = self._upsample_to(disp_i, full_hw)
                outputs[("disp", 0)] = disp_full

        return outputs
