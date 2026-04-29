import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import os
from typing import Optional

# ========= 多尺度封装（默认强制加载权重）=========
# 1) arch → 默认本地权重路径映射
_DINO_WEIGHTS = {
    "dinov3_convnext_tiny":  "/mnt/data_nvme3n1p1/PycharmProjects/dinov3/checkpoints/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "dinov3_convnext_small": "/mnt/data_nvme3n1p1/PycharmProjects/dinov3/checkpoints/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
}


def _extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module", "teacher"):
            nested = checkpoint.get(key, None)
            if isinstance(nested, dict):
                return nested
    return checkpoint


def _count_matched_tensors(module_state, checkpoint_state):
    normalized = {}
    for k, v in checkpoint_state.items():
        if not hasattr(v, "shape"):
            continue
        for nk in (
            k,
            k[7:] if k.startswith("module.") else k,
            k[9:] if k.startswith("backbone.") else k,
            k[6:] if k.startswith("model.") else k,
        ):
            if nk not in normalized:
                normalized[nk] = v
    matched = 0
    for k, v in module_state.items():
        vv = normalized.get(k, None)
        if vv is not None and hasattr(vv, "shape") and vv.shape == v.shape:
            matched += 1
    return matched, len(module_state)


class DinoConvNeXtMultiScale(nn.Module):
    def __init__(
        self,
        repo_dir: str = "/mnt/data_nvme3n1p1/PycharmProjects/dinov3",
        arch: str = "dinov3_convnext_tiny",
        # 若传入 "auto" 或 None，则按 arch 自动匹配本地权重；也可直接给路径覆盖
        weights: Optional[str] = "auto",
        source: str = "local",
        out_indices=(0, 1, 2, 3)
    ):
        super().__init__()

        # 解析权重
        if weights is None or (isinstance(weights, str) and weights.lower() == "auto"):
            if arch not in _DINO_WEIGHTS:
                raise KeyError(f"未配置 arch 的默认权重：{arch}")
            weights_path = _DINO_WEIGHTS[arch]
        else:
            weights_path = weights

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"权重文件不存在：{weights_path}")

        self.backbone = torch.hub.load(repo_dir, arch, source=source, weights=weights_path)
        self.pretrained_weights_path = os.path.abspath(weights_path)
        self.pretrained_num_loaded = None
        self.pretrained_num_total = None
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            checkpoint_state = _extract_checkpoint_state_dict(checkpoint)
            if isinstance(checkpoint_state, dict):
                matched_num, total_num = _count_matched_tensors(self.backbone.state_dict(), checkpoint_state)
                self.pretrained_num_loaded = matched_num
                self.pretrained_num_total = total_num
        except Exception:
            # 统计失败不影响主流程，保持原加载行为。
            pass
        self.out_indices = tuple(out_indices)
        # 与特征列表一一对应
        self.num_ch_enc = np.array([96, 192, 384, 768], dtype=np.int64)

    def forward(self, x):
        outs = []
        cur = x
        for i in range(4):
            cur = self.backbone.downsample_layers[i](cur)
            cur = self.backbone.stages[i](cur)
            if i in self.out_indices:
                outs.append(cur)  # 按 i=0..3 顺序依次追加：C1, C2, C3, C4
        return outs  # ★ 返回 list[Tensor]，最后一个元素是最低分辨率


if __name__ == "__main__":
    # 设备（也可不传，类内部会自动判定）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化：将在内部按默认路径强制加载权重
    backbone_ms = DinoConvNeXtMultiScale()
    backbone_ms = backbone_ms.to(device).eval()

    # ========= 预处理与测试 =========
    transform = T.Compose([
        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_path = "/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_images/DJI_0168/image_02/data/0000000001.jpg"
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1,3,224,224]

    with torch.no_grad():  # 推理；训练时去掉
        feats = backbone_ms(x)

    for k in ["C1", "C2", "C3", "C4"]:
        v = feats[k]
        print(f"{k}: {tuple(v.shape)}")
