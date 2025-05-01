import torch
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image

# 1. Setup Pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
).to(device)

# 2. Load LoRA weights - CORRECTED METHOD
lora_path = " /home/ubuntu/money-leads-video-generator/peter_model/final_model_peter"
pipe.unet.load_attn_procs(lora_path)

# 3. Generate Frames
num_frames = 24
frames = []
print("Generating animation frames...")

for i in range(num_frames):
    # Progressive motion prompts
    progress = i / num_frames
    if progress < 0.3:
        action = "starting to clap"
    elif progress < 0.7:
        action = "clapping hands together"
    else:
        action = "finishing clap motion"
    
    prompt = f"a person {action}, high quality, detailed, 4k resolution"
    
    with torch.autocast(device):
        image = pipe(
            prompt=prompt,
            negative_prompt="blurry, deformed, low quality, bad anatomy",
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
    
    frames.append(np.array(image))
    print(f"Generated frame {i+1}/{num_frames}")

# 4. Create MP4 Video
print("Creating MP4 video...")
height, width, _ = frames[0].shape
video = cv2.VideoWriter(
    'clapping_video.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'),
    12,  # FPS
    (width, height)
)

for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
print("Successfully created clapping_video.mp4")
