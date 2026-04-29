import os
from PIL import Image

def center_crop_4096_to_3840(img):
    """如果图像是 4096x2160，中心裁剪成 3840x2160，并返回裁剪标志"""
    if img.size == (4096, 2160):
        left = (4096 - 3840) // 2
        upper = 0
        right = left + 3840
        lower = 2160
        return img.crop((left, upper, right, lower)), True
    return img, False

def resize_preserve_aspect(img, max_size):
    """按比例缩放图像，使其最大不超过 max_size"""
    original_width, original_height = img.size
    max_width, max_height = max_size

    ratio = min(max_width / original_width, max_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    return img.resize(new_size, Image.BILINEAR)

def resize_and_save_png(input_path, output_path, max_size):
    os.makedirs(output_path, exist_ok=True)
    for file_name in os.listdir(input_path):
        if file_name.lower().endswith(".jpg"):
            input_file = os.path.join(input_path, file_name)
            output_file = os.path.join(output_path, file_name)

            try:
                img = Image.open(input_file)
                img, cropped = center_crop_4096_to_3840(img)
                if cropped:
                    print(f"已中心裁剪: {input_file}")

                img = resize_preserve_aspect(img, max_size)
                img.save(output_file, "PNG")
            except Exception as e:
                print(f"处理失败: {input_file}，错误: {e}")

def process_dataset(root_dir, max_size, output_subdir_name):
    for split in ["Train", "Validation"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for seq_name in os.listdir(split_path):
            seq_path = os.path.join(split_path, seq_name)
            data_path = os.path.join(seq_path, "data")
            output_path = os.path.join(seq_path, output_subdir_name)

            if os.path.isdir(data_path):
                print(f"正在处理：{data_path}")
                resize_and_save_png(data_path, output_path, max_size)
            else:
                print(f"未找到data目录：{data_path}")

if __name__ == "__main__":
    root_dataset_dir = "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany"
    max_resize_size = (1024, 576)                      # 可配置缩放尺寸
    output_subdir_name = "data_1280"                   # 可配置输出子目录名
    process_dataset(root_dataset_dir, max_resize_size, output_subdir_name)
