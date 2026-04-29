import os
import cv2

# 设置根路径（请替换为您的实际路径）
#root_dir = "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China"
root_dir = "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China/Validation/sfm"
#root_dir = "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany"
target_folders = ["Train", "Validation", "Test"]

# 提取帧速率（每秒几帧）China frame +-5
extract_fps = 5
# 提取帧速率（每秒几帧）Germany frame +-10
#extract_fps = 10

def extract_frames_from_video(video_path, output_dir, fps=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频：{video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"[错误] 视频帧率为0：{video_path}")
        return

    interval = int(round(video_fps / fps))
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            filename = os.path.join(output_dir, f"{saved_idx:010d}.jpg")
            cv2.imwrite(filename, frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"[完成] {video_path} → 共提取 {saved_idx} 帧到 {output_dir}")

# 遍历每个阶段文件夹（Train / Validation / Test）
for stage in target_folders:
    stage_path = os.path.join(root_dir, stage)
    if not os.path.isdir(stage_path):
        print(f"[警告] 跳过不存在的目录：{stage_path}")
        continue

    for seq in sorted(os.listdir(stage_path)):
        seq_path = os.path.join(stage_path, seq)
        if not os.path.isdir(seq_path):
            continue

        # 查找 .mp4 文件
        for file in os.listdir(seq_path):
            if file.lower().endswith(".mp4"):
                video_path = os.path.join(seq_path, file)
                output_dir = os.path.join(seq_path, "data")
                extract_frames_from_video(video_path, output_dir, extract_fps)
