# Load the base model
import os
import torch
from flask import Flask, request, send_file, send_from_directory
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from lora_diffusion import patch_pipe
import numpy as np
import tempfile
from flask_cors import CORS
import cv2

app = Flask(__name__, static_folder="static")
CORS(app)

# Model paths
model_path = "/home/ubuntu/money-leads-video-generator/peter_model2"
base_model = "runwayml/stable-diffusion-v1-5"

# Initialize pipeline
logger.info("Loading base model...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

# Load LoRA weights
lora_path = os.path.join(model_path, "unet_lora_weights.bin")
if os.path.exists(lora_path):
    logger.info("Applying LoRA weights...")
    patch_pipe(
        pipe,
        lora_path,
        patch_text=False  # We're only using UNet LoRA in this example
    )
else:
    logger.error(f"LoRA weights not found at {lora_path}")
    raise FileNotFoundError(f"LoRA weights not found at {lora_path}")

# Move to GPU and enable optimizations
pipe.to("cuda")
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
        num_frames = 8
        frames = []

        for i in range(num_frames):
            strength = 0.5 + (i * 0.05)
            
            result = pipe(
                prompt=prompt,
                image=img,
                num_inference_steps=30,
                strength=min(strength, 0.9),
                guidance_scale=7.5
            )
            
            frame = np.array(result.images[0])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)

        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name
            
        # Write video
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
