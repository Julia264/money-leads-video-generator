from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import export_to_video
import torch
import tempfile
import os

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Initialize the pipeline with the base model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16"
)

# Load your custom UNet model
custom_unet_path = "/home/ubuntu/money-leads-video-generator/peter_model/final_model_peter/unet"
try:
    # Try loading as a full model first
    pipe.unet = UNet2DConditionModel.from_pretrained(
        custom_unet_path,
        torch_dtype=torch.float16
    )
except:
    # Fall back to trying to load just LoRA weights
    pipe.unet.load_attn_procs(
        custom_unet_path,
        use_safetensors=False  # Set to True if using .safetensors format
    )

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Enable memory efficient attention if available
if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    pipe.enable_xformers_memory_efficient_attention()

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/generate-video", methods=["POST"])
def generate_video():
    temp_path = None
    try:
        # Get the uploaded image
        image_file = request.files["image"]
        img = Image.open(image_file).convert("RGB")
        
        # Resize image to expected dimensions
        img = img.resize((1024, 576))
        
        # Generate video from the image
        frames = pipe(
            image=img,
            decode_chunk_size=4,  # Reduced for memory constraints
            motion_bucket_id=180,
            noise_aug_strength=0.1,
            num_frames=25,
            num_inference_steps=25
        ).frames[0]
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Export frames to video
        export_to_video(frames, temp_path, fps=7)
        
        return send_file(
            temp_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="generated_video.mp4"
        )
        
    except Exception as e:
        return {"error": str(e)}, 500
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
