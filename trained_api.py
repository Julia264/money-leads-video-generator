import os
import torch
from flask import Flask, request, send_file, send_from_directory
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import tempfile
from flask_cors import CORS
from moviepy.editor import ImageSequenceClip

# Initialize Flask
app = Flask(__name__, static_folder="static")
CORS(app)

# Load the trained model
model_path = "./peter_model/final_model_peter"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
).to("cuda")

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
            frames.append(frame)

        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            clip = ImageSequenceClip(frames, fps=4)
            clip.write_videofile(tmp_file.name, codec="libx264", audio=False)
            video_path = tmp_file.name

        return send_file(video_path, mimetype="video/mp4")

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
