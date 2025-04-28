from PIL import Image
import os

frames_dir = "./datasets/frames"

for folder in os.listdir(frames_dir):
    folder_path = os.path.join(frames_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(file_path).convert("RGB")
                    img.verify()  # Verify image is not corrupt
                except Exception as e:
                    print(f"Corrupted file: {file_path} ({e})")
