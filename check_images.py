import os
from PIL import Image
import shutil

def clean_dataset(root_dir, image_size=(512, 512)):
    valid_images = []
    corrupted_images = []
    
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        img = Image.open(file_path)
                        img.verify()  # Verify if image can be opened
                        img = Image.open(file_path).convert("RGB")  # Reload properly
                        if img.size[0] < 256 or img.size[1] < 256:
                            # Too small to be useful
                            corrupted_images.append(file_path)
                        else:
                            valid_images.append(file_path)
                    except Exception as e:
                        corrupted_images.append(file_path)
    
    # حذف الصور التالفة
    for path in corrupted_images:
        os.remove(path)
    
    print(f"✅ Cleaned dataset in: {root_dir}")
    print(f"🟢 Valid images kept: {len(valid_images)}")
    print(f"🔴 Corrupted or bad images deleted: {len(corrupted_images)}")

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    data_dir = os.path.join(BASE_DIR, "datasets", "frames")  # مكان ملفاتك

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")

    clean_dataset(data_dir)
