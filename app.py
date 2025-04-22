from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
import os
app = Flask(__name__, static_url_path='/static')
CORS(app)

# Load the pipeline once
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset for video frames
class VideoFramesDataset(Dataset):
    def __init__(self, frame_dir, transform=None):
        self.frame_dir = frame_dir
        self.transform = transform
        self.frames = []
        for video_folder in os.listdir(frame_dir):
            video_folder_path = os.path.join(frame_dir, video_folder)
            if os.path.isdir(video_folder_path):
                frames_in_video = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(".png")]
                self.frames.extend(frames_in_video)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_name = self.frames[idx]
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Define the transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset and dataloader
frame_dir = "/home/ubuntu/money-leads-video-generator/svd-env/datasets/frames"
dataset = VideoFramesDataset(frame_dir=frame_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Fine-tuning function
def fine_tune_model():
    optimizer = AdamW(pipe.parameters(), lr=1e-5)

    epochs = 10
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = pipe(batch)
            
            # Define loss calculation based on the specific task and model's forward output
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()}")

    # Save the fine-tuned model
    pipe.save_pretrained("path_to_save_model")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/generate-video", methods=["POST"])
def generate_video():
    # Get the uploaded image
    image_file = request.files["image"]
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)

    # Convert image to tensor and normalize it
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).permute(0, 3, 1, 2)
    img_tensor = img_tensor.to(torch.float32)

    # Generate video frames
    video_frames = pipe(img_tensor, num_frames=8).frames[0]
    video_frames = [np.array(frame) for frame in video_frames]

    # Create video clip
    clip = ImageSequenceClip(video_frames, fps=30)
    clip = clip.with_duration(5)

    # Export video in high quality
    # Export video in high quality
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        #clip = clip.resize(height=1080, width=1920)  # upscale after generation
        clip.write_videofile(
            temp.name,
            codec="libx264",
            bitrate="5000k",
            fps=30,
            preset="slow",
            audio=False
        )

        return send_file(
            temp.name,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="output_with_motion.mp4"
        )
@app.route("/fine-tune", methods=["POST"])
def fine_tune():
    fine_tune_model()
    return "Model fine-tuning completed!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
