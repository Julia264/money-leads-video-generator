import torch
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# 1. Verify and set up paths
lora_path = "/home/ubuntu/money-leads-video-generator/peter_model/final_model_peter"
output_dir = "/home/ubuntu/money-leads-video-generator/output_videos"
os.makedirs(output_dir, exist_ok=True)

# Verify LoRA files exist
required_files = ["pytorch_lora_weights.bin", "config.json"]
for file in required_files:
    if not os.path.exists(os.path.join(lora_path, file)):
        raise FileNotFoundError(f"Missing required file: {os.path.join(lora_path, file)}")

# 2. Initialize pipeline with proper LoRA loading
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
).to(device)

# Load LoRA weights - METHOD 1 (Preferred)
try:
    pipe.unet.load_attn_procs(lora_path)
except Exception as e:
    print(f"Method 1 failed: {str(e)}")
    # Fallback to METHOD 2
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        lora_path,
        torch_dtype=torch.float16,
        subfolder="unet"
    )
    pipe.unet = unet.to(device)

# 3. Generate frames with proper motion progression
num_frames = 24
height, width = 512, 512
frames = []

print("\nGenerating animation frames...")
for i in range(num_frames):
    # Calculate progress through motion
    progress = i / num_frames
    
    # Define motion stages
    if progress < 0.25:
        action = "raising hands to begin clapping"
    elif progress < 0.5:
        action = "bringing hands together to clap"
    elif progress < 0.75:
        action = "hands clapping together"
    else:
        action = "lowering hands after clapping"
    
    prompt = f"a person {action}, highly detailed, 4k resolution, professional photography"
    negative_prompt = "blurry, deformed, low quality, bad anatomy, extra limbs"
    
    # Generate frame
    with torch.autocast(device):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
    
    frames.append(np.array(image))
    print(f"Generated frame {i+1}/{num_frames} - {action}")

# 4. Create MP4 video with proper encoding
video_path = os.path.join(output_dir, "clapping_animation.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, 12.0, (width, height))

for frame in frames:
    # Convert RGB to BGR and write frame
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
print(f"\nSuccessfully created video at: {video_path}")
