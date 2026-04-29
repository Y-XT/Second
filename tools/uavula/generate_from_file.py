import os
from typing import List

def generate_kitti_style_list(base_dir: str, subset: str, output_file: str, frame_ids: List[int], data_folder_name: str):
    """
    生成单个子集的 KITTI 风格图像列表文件。
    跳过每个序列前后 max(|frame_ids|) 帧，不检查帧是否连续。
    """
    subset_dir = os.path.join(base_dir, subset)
    if not os.path.isdir(subset_dir):
        print(f"❌ 无效子集路径: {subset_dir}")
        return

    sequence_dirs = sorted(os.listdir(subset_dir))
    print(f"在 {subset} 中发现的序列目录：")
    for seq in sequence_dirs:
        print(f"  - {seq}")

    with open(output_file, "w") as f:
        for seq in sequence_dirs:
            data_path = os.path.join(subset, seq, data_folder_name)  # 相对路径写入
            abs_data_path = os.path.join(base_dir, data_path)
            if not os.path.isdir(abs_data_path):
                print(f"[跳过] 未找到 data 文件夹：{abs_data_path}")
                continue

            img_list = sorted(fn for fn in os.listdir(abs_data_path) if fn.lower().endswith(".jpg"))
            if not img_list:
                print(f"[跳过] 空图像目录：{abs_data_path}")
                continue

            max_offset = max(abs(i) for i in frame_ids)
            valid_imgs = img_list[max_offset:len(img_list) - max_offset]

            for img_name in valid_imgs:
                stem = os.path.splitext(img_name)[0].lstrip("0")
                try:
                    frame_index = int(stem) if stem else 0
                except ValueError:
                    continue  # 跳过非法命名文件
                f.write(f"{data_path} {frame_index}\n")

            print(f"[完成] {data_path}：记录 {len(valid_imgs)} 帧")

    print(f"✅ 已生成：{output_file}")


def generate_all_subsets(base_dir: str, frame_ids: List[int], data_folder_name: str):
    """
    处理 base_dir 下所有子集目录，输出文件命名为：
    train_files.txt / val_files.txt / test_files.txt，保存在当前运行目录。
    """
    subsets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    subset_name_map = {
        "Train": "train_files.txt",
        "Validation": "val_files.txt",
        "Val": "val_files.txt",
        "Test": "test_files.txt"
    }

    print(f"\n📍 开始处理数据根目录：{base_dir}")
    for subset in subsets:
        output_name = subset_name_map.get(subset, f"{subset.lower()}_files.txt")
        output_file = os.path.join(os.getcwd(), output_name)  # 保存到当前运行目录
        generate_kitti_style_list(
            base_dir=base_dir,
            subset=subset,
            output_file=output_file,
            frame_ids=frame_ids,
            data_folder_name=data_folder_name
        )


if __name__ == "__main__":
    base_dir = "/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset"
    frame_ids = [-1, 0, 1]
    data_folder_name = "image_02/data"  # <-- 可替换为实际使用的数据子目录名

    generate_all_subsets(base_dir=base_dir, frame_ids=frame_ids, data_folder_name=data_folder_name)
