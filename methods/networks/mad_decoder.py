
from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from methods.losses.Mad.layers_3d import *


class DepthDecoder_3d(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_3d, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        # self.num_ch_enc = np.array([64,64,128,256,512])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0  ConvBlock contain Conv3x3 includes Conv3d
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3_2d(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                # Collapse temporal dim once near output.
                x2d = torch.mean(x, dim=2)
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x2d))
        return self.outputs

if __name__ == "__main__":

    B, H, W = 2, 96, 96
    num_ch_enc = [64, 64, 128, 256, 512]
    scales = [0, 1, 2, 3]

    # Simulate encoder output for two images
    features1 = [
        torch.randn(B, c, H // (2 ** (i + 1)), W // (2 ** (i + 1))) for i, c in enumerate(num_ch_enc)
    ]
    features2 = [
        torch.randn_like(f) for f in features1
    ]

    # Stack them into [B, C, 2, H, W] format
    fused_features = [
        torch.stack([f1, f2], dim=2) for f1, f2 in zip(features1, features2)
    ]

    # Run decoder
    decoder = DepthDecoder_3d(num_ch_enc=num_ch_enc, scales=scales)
    outputs = decoder(fused_features)

    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
