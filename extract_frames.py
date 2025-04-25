import os
import cv2
import unicodedata
import shutil

VIDEO_DIR = r"/home/ubuntu/money-leads-video-generator/svd-env/datasets/الحركات"
OUTPUT_DIR = r"/home/ubuntu/money-leads-video-generator/svd-env/datasets/frames"

def normalize_name(name):
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").replace(" ", "_")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for video_file in os.listdir(VIDEO_DIR):
    if video_file.endswith(".mp4") or video_file.endswith(".mov"):
        label = os.path.splitext(video_file)[0]
        label_safe = normalize_name(label)
        label_folder = os.path.join(OUTPUT_DIR, label_safe)
        os.makedirs(label_folder, exist_ok=True)

        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            continue

        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_filename = os.path.join(label_folder, f"frame_{frame_count:03d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()

        # Optionally rename the folder back to Arabic
        arabic_path = os.path.join(OUTPUT_DIR, label)
        if label_safe != label:
            try:
                shutil.move(label_folder, arabic_path)
            except:
                pass

        print(f"✅ {video_file} → {frame_count} frames extracted to dataset/{label}")
