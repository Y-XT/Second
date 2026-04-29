from .mono_dataset import MonoDataset
import numpy as np
import random
import torch
from torchvision import transforms
import PIL.Image as pil
import os

class WildUAVDataset(MonoDataset):
    """WildUAV数据集加载器
    
    Args:
        data_path: 数据集根目录路径
        filenames: 文件名列表，格式为 "video.7z/vidXX/vidXX_XXX frame_index"
        height: 图像高度
        width: 图像宽度
        frame_idxs: 帧索引列表，默认为[0, -1, 1]
        num_scales: 金字塔层数，默认为4
        is_train: 是否为训练模式
        img_ext: 图像文件扩展名
        normalize: 是否启用归一化
        norm_mode: 归一化模式 ("imagenet" 或 "custom")
        norm_mean: 自定义归一化均值
        norm_std: 自定义归一化标准差
    """
    def __init__(self, *args,
                 normalize=False,               # 开关：是否启用归一化
                 norm_mode="imagenet",          # "imagenet" 或 "custom"
                 norm_mean=(0.485, 0.456, 0.406),
                 norm_std=(0.229, 0.224, 0.225),
                 **kwargs):
        super(WildUAVDataset, self).__init__(*args, **kwargs)

        # 归一化配置
        self.normalize = bool(normalize)
        if norm_mode == "imagenet":
            self._norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self._norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif norm_mode == "custom":
            self._norm_mean = np.array(norm_mean, dtype=np.float32)
            self._norm_std = np.array(norm_std, dtype=np.float32)
        else:
            raise ValueError("norm_mode must be 'imagenet' or 'custom'")

        # WildUAV数据集相机内参矩阵
        # 基于4K分辨率 (3840x2160) 的假设，主点位于图像中心
        # 焦距基于典型无人机相机的视场角估算
        self.K = np.array([[0.8615367073, 0, 0.5013700612, 0],
                           [0, 1.1498771017, 0.4964614050, 0],  # 调整fy以适应16:9比例
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.missing_images = []
        self.full_res_shape = (3840, 2160)  # WildUAV数据集的实际分辨率

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps.

        <frame_id> 是一个整数（例如 0、-1 或 1），表示相对于当前样本的时间步。

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and self.allow_flip and random.random() > 0.5

        # 解析WildUAV数据集的文件路径格式
        # 格式: "video.7z/vidXX/vidXX_XXX frame_index"
        line = self.filenames[index].split()
        if len(line) != 2:
            # 如果格式不正确，尝试下一个样本
            return self.__getitem__((index + 1) % len(self.filenames))
        
        folder_path = line[0]  # "video.7z/vidXX/vidXX_XXX"
        frame_index = int(line[1])

        for i in self.frame_idxs:
            target_index = frame_index + i
            # 构建图像文件名，格式: vidXX_XXX_XXXXX.jpg
            # 需要从folder_path中提取视频ID和子序列ID
            path_parts = folder_path.split('/')
            if len(path_parts) >= 3:
                vid_id = path_parts[1]  # vidXX
                sub_seq_id = path_parts[2]  # vidXX_XXX
                fname = f"{sub_seq_id}_{target_index:05d}{self.img_ext}"
                image_path = os.path.join(self.data_path, folder_path, fname)
            else:
                # 如果路径格式不正确，跳过这个样本
                return self.__getitem__((index + 1) % len(self.filenames))

            if not os.path.exists(image_path):
                # 如果图像不存在，尝试下一个样本
                return self.__getitem__((index + 1) % len(self.filenames))

            color = self.loader(image_path)

            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)

            inputs[("color", i, -1)] = color

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # 添加归一化支持
        if self.normalize:
            mean = torch.tensor(self._norm_mean, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(self._norm_std, dtype=torch.float32).view(3, 1, 1)

            for i in self.frame_idxs:
                for s in range(self.num_scales):
                    for prefix in ("color", "color_aug"):
                        key = (prefix, i, s)
                        if key not in inputs:
                            continue
                        t = inputs[key]
                        # 确保数据类型和范围正确
                        if t.dtype != torch.float32:
                            t = t.float()
                        if t.max() > 1.0:
                            t = t / 255.0
                        mm = mean.to(t.device)
                        ss = std.to(t.device)
                        inputs[key] = (t - mm) / ss

        if self.load_depth:
            # WildUAV数据集可能没有深度图，这里保留接口
            # depth_gt = self.get_depth(folder_path, frame_index, do_flip)
            # inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            # inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
            pass

        return inputs

    def check_depth(self):
        """检查WildUAV数据集是否有深度图"""
        # WildUAV数据集通常不提供深度图，返回False
        return False
