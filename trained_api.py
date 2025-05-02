import os
import torch
from flask import Flask, request, send_file, send_from_directory
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import tempfile
from flask_cors import CORS
import cv2

# Initialize Flask
app = Flask(__name__, static_folder="static")
CORS(app)

# Load the trained model
model_path = "./peter_model/final_model_peter"

# First load the pipeline to CPU
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

# Then move to GPU
pipe.to("cuda")

# Enable optimizations
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/generate-video", methods=["POST"])
def generate_video():
    if "image" not in request.files:
        return {"error": "No image uploaded"}, 400

    try:
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((512, 512))

        prompt = "a person clapping hands"
        num_frames = 8  # Number of frames for the video
        frames = []

        for i in range(num_frames):
            strength = 0.5 + (i * 0.05)  # Gradually increase strength
            
            result = pipe(
                prompt=prompt,
                image=img,
                num_inference_steps=30,
                strength=min(strength, 0.9),  # Cap strength at 0.9
                guidance_scale=7.5
            )
            
            frame = np.array(result.images[0])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            frames.append(frame)

        # Create temporary video file with OpenCV
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name
            
        # Write video using OpenCV
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 4, (width, height))
        
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

        return send_file(video_path, mimetype="video/mp4")

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
