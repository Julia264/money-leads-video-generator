import os
import torch
from flask import Flask, request, send_file, send_from_directory, jsonify
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from lora_diffusion import patch_pipe
import numpy as np
import tempfile
from flask_cors import CORS
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "/home/ubuntu/money-leads-video-generator/peter_model2")
BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
OUTPUT_SIZE = (512, 512)
FRAME_COUNT = 8
FPS = 4

def load_model():
    """Load the model with optional LoRA weights"""
    try:
        logger.info(f"Loading base model: {BASE_MODEL}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        # Try to find LoRA weights
        lora_path = None
        for filename in ["unet_lora_weights.bin", "pytorch_lora_weights.bin", "lora_weights.bin"]:
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                lora_path = path
                break

        if lora_path:
            logger.info(f"Applying LoRA weights from {lora_path}")
            patch_pipe(pipe, lora_path, patch_text=False)
        else:
            logger.warning("No LoRA weights found, using base model only")

        # Optimizations
        if torch.cuda.is_available():
            pipe.to("cuda")
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            pipe.enable_attention_slicing()
        
        return pipe

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

pipe = load_model()

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/generate-video", methods=["POST"])
def generate_video():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Process input image
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize(OUTPUT_SIZE)

        # Generate frames
        frames = []
        for i in range(FRAME_COUNT):
            strength = 0.5 + (i * 0.05)
            result = pipe(
                prompt="a person clapping hands",
                image=img,
                num_inference_steps=30,
                strength=min(strength, 0.9),
                guidance_scale=7.5
            )
            frame = np.array(result.images[0])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)

        # Create video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path, 
                fourcc, 
                FPS, 
                OUTPUT_SIZE
            )
            
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

            # Read the video file back to ensure it's valid
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise ValueError("Generated video file is empty")

            return send_file(
                video_path,
                mimetype="video/mp4",
                as_attachment=True,
                download_name="generated_video.mp4"
            )

    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting server...")
    app.run(host="0.0.0.0", port=8000, debug=False)  # Disable debug for production
