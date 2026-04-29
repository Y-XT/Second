import zipfile
import os

zip_dir = "/home/yxt/PycharmProjects/dataset/kitti_raw"
for fname in os.listdir(zip_dir):
    if fname.endswith(".zip"):
        fpath = os.path.join(zip_dir, fname)
        try:
            with zipfile.ZipFile(fpath, 'r') as zf:
                bad_file = zf.testzip()
                if bad_file:
                    print(f"[BAD CONTENT] {fname} → corrupted entry: {bad_file}")
        except zipfile.BadZipFile:
            print(f"[CORRUPT FILE] {fname} → not a valid zip")
        except Exception as e:
            print(f"[ERROR] {fname} → {e}")
