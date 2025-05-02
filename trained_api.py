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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# Model paths - make these configurable via environment variables
MODEL_DIR = os.getenv("MODEL_DIR", "/home/ubuntu/money-leads-video-generator/peter_model2")
BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")

def load_model_with_lora():
    """Load the base model and apply LoRA weights"""
    try:
        logger.info(f"Loading base model: {BASE_MODEL}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        # Try multiple possible locations for LoRA weights
        possible_lora_paths = [
            os.path.join(MODEL_DIR, "unet_lora_weights.bin"),
            os.path.join(MODEL_DIR, "pytorch_lora_weights.bin"),
            os.path.join(MODEL_DIR, "lora_weights.bin")
        ]

        lora_path = None
        for path in possible_lora_paths:
            if os.path.exists(path):
                lora_path = path
                break

        if lora_path:
            logger.info(f"Applying LoRA weights from {lora_path}")
            patch_pipe(
                pipe,
                lora_path,
                patch_text=False
            )
            logger.info("Successfully loaded LoRA weights")
        else:
            logger.warning("No LoRA weights found, using base model only")

        # Move to GPU and enable optimizations
        if torch.cuda.is_available():
            logger.info("Moving model to GPU...")
            pipe.to("cuda")
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            pipe.enable_attention_slicing()
        else:
            logger.warning("CUDA not available, using CPU")

        return pipe

    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

# Load the model when starting the app
try:
    pipe = load_model_with_lora()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/generate-video", methods=["POST"])
def generate_video():
    if "image" not in request.files:
        logger.error("No image uploaded in request")
        return {"error": "No image uploaded"}, 400

    try:
        file = request.files["image"]
        logger.info("Processing image...")
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((512, 512))

        prompt = "a person clapping hands"
        num_frames = 8
        frames = []

        logger.info(f"Generating {num_frames} frames...")
        for i in range(num_frames):
            strength = 0.5 + (i * 0.05)
            logger.debug(f"Generating frame {i+1} with strength {min(strength, 0.9)}")
            
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

        logger.info("Video generation complete")
        return send_file(video_path, mimetype="video/mp4")

    except Exception as e:
        logger.error(f"Error during video generation: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000, debug=True)
