from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import ResnetEncoder


class SPIUpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        fused = torch.cat([up_x, concat_with], dim=1)
        return self.net(fused)


class SPIFeatureDecoderBN(nn.Module):
    def __init__(self, encoder_channels, num_features=512, num_classes=32):
        super().__init__()
        features = int(num_features)
        bottleneck_features = int(encoder_channels[-1])

        # Keep the same layer layout as SPIdepth so external checkpoints match.
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.up1 = SPIUpSampleBN(skip_input=features + int(encoder_channels[-2]), output_features=features // 2)
        self.up2 = SPIUpSampleBN(skip_input=features // 2 + int(encoder_channels[-3]), output_features=features // 4)
        self.up3 = SPIUpSampleBN(skip_input=features // 4 + int(encoder_channels[-4]), output_features=features // 8)
        self.up4 = SPIUpSampleBN(skip_input=features // 8 + int(encoder_channels[-5]), output_features=features // 16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class SPIResnetEncoderDecoder(nn.Module):
    def __init__(self, num_layers=50, num_features=512, model_dim=32, pretrained=True):
        super().__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers,
            pretrained=pretrained,
            num_input_images=1,
        )
        self.pretrained_num_loaded = getattr(self.encoder, "pretrained_num_loaded", None)
        self.pretrained_num_total = getattr(self.encoder, "pretrained_num_total", None)
        self.decoder = SPIFeatureDecoderBN(
            encoder_channels=self.encoder.num_ch_enc,
            num_features=num_features,
            num_classes=model_dim,
        )

    def forward(self, x, **kwargs):
        features = self.encoder(x)
        return self.decoder(features, **kwargs)


class SPILiteResnetEncoderDecoder(nn.Module):
    def __init__(self, model_dim=128, pretrained=True):
        super().__init__()
        self.encoder = ResnetEncoder(
            num_layers=18,
            pretrained=pretrained,
            num_input_images=1,
        )
        self.pretrained_num_loaded = getattr(self.encoder, "pretrained_num_loaded", None)
        self.pretrained_num_total = getattr(self.encoder, "pretrained_num_total", None)
        self.decoder = SPIFeatureDecoderBN(
            encoder_channels=self.encoder.num_ch_enc,
            num_features=256,
            num_classes=model_dim,
        )

    def forward(self, x, **kwargs):
        features = self.encoder(x)
        return self.decoder(features, **kwargs)


class SPIFullQueryLayer(nn.Module):
    def forward(self, x, queries):
        n, c, h, w = x.size()
        _, num_queries, query_dim = queries.size()
        if c != query_dim:
            raise ValueError(
                f"Feature channels ({c}) must match query dim ({query_dim}) for SPIdepth full query attention"
            )

        attn = torch.matmul(
            x.view(n, c, h * w).permute(0, 2, 1),
            queries.permute(0, 2, 1),
        )
        attn_norm = torch.softmax(attn, dim=1)
        summary = torch.matmul(
            attn_norm.permute(0, 2, 1),
            x.view(n, c, h * w).permute(0, 2, 1),
        )
        energy_maps = attn.permute(0, 2, 1).view(n, num_queries, h, w)
        return energy_maps, summary


class SPIDepthDecoderQueryTr(nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_dim=128,
        patch_size=16,
        num_heads=4,
        query_nums=100,
        dim_out=256,
        transformer_ff_dim=1024,
        norm="linear",
        min_val=0.1,
        max_val=100.0,
    ):
        super().__init__()
        self.norm = norm
        self.query_nums = int(query_nums)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        self.embedding_convPxP = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            num_heads,
            dim_feedforward=transformer_ff_dim,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.full_query_layer = SPIFullQueryLayer()
        self.bins_regressor = nn.Sequential(
            nn.Linear(embedding_dim * self.query_nums, 16 * self.query_nums),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16 * self.query_nums, 16 * 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16 * 16, dim_out),
        )
        self.convert_to_prob = nn.Sequential(
            nn.Conv2d(self.query_nums, dim_out, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

    def forward(self, x0):
        embeddings = self.embedding_convPxP(x0.clone())
        embeddings = embeddings.flatten(2)
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)
        embeddings = embeddings.permute(2, 0, 1)
        total_queries = self.transformer_encoder(embeddings)

        x0 = self.conv3x3(x0)
        queries = total_queries[:self.query_nums, ...].permute(1, 0, 2)

        energy_maps, summaries = self.full_query_layer(x0, queries)
        batch_size, query_count, embed_dim = summaries.shape
        bins = self.bins_regressor(summaries.view(batch_size, query_count * embed_dim))

        if self.norm == "linear":
            bins = torch.relu(bins) + 0.1
        elif self.norm == "softmax":
            return torch.softmax(bins, dim=1), energy_maps
        else:
            bins = torch.sigmoid(bins)

        bins = bins / bins.sum(dim=1, keepdim=True)
        prob = self.convert_to_prob(energy_maps)
        bin_widths = (self.max_val - self.min_val) * bins
        bin_widths = F.pad(bin_widths, (1, 0), mode="constant", value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        centers = centers.view(centers.size(0), centers.size(1), 1, 1)
        depth = torch.sum(prob * centers, dim=1, keepdim=True)

        outputs = {}
        # Match official SPIdepth semantics: ('disp', 0) stores depth values directly.
        outputs[("disp", 0)] = depth
        outputs[("spi_depth", 0)] = depth
        return outputs
