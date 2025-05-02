import os
import torch
from flask import Flask, request, send_file, send_from_directory
from PIL import Image
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageSequenceClip
import numpy as np
import tempfile
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, static_folder="static")
CORS(app)

# Load trained model
model_path = "./peter_model/final_model_peter"
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    safety_checker=None,
    requires_safety_checker=False,
).to("cuda")

# Optional: enable performance settings
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

    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    img = img.resize((512, 512))

    prompt = "a person clapping hands"

    # Generate multiple frames by sampling with different seeds
    frames = []
    for i in range(30):
        generator = torch.manual_seed(i)  # Vary the seed per frame
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=img,
                num_inference_steps=30,
                strength=0.7,
                guidance_scale=7.5,
                generator=generator
            )
        frame = np.array(result.images[0])
        frames.append(frame)

    # Save frames into a temporary MP4 video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        clip = ImageSequenceClip(frames, fps=8)
        clip.write_videofile(tmp_file.name, codec="libx264", audio=False)
        return send_file(tmp_file.name, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
