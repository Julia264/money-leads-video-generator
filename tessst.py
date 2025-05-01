import torch
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# 1. Setup paths
model_dir = "/home/ubuntu/money-leads-video-generator/peter_model/final_model_peter"
output_dir = "/home/ubuntu/money-leads-video-generator/output_videos"
os.makedirs(output_dir, exist_ok=True)

# 2. Load the model properly with device placement
print("Loading trained model...")

# First load to CPU to avoid meta tensor issues
pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,  # Start with float32 on CPU
    safety_checker=None
)

# Then move to GPU if available
if torch.cuda.is_available():
    print("Moving model to GPU...")
    pipe = pipe.to("cuda")
    # Convert to float16 if on GPU
    pipe.unet = pipe.unet.half()
    pipe.vae = pipe.vae.half()
    pipe.text_encoder = pipe.text_encoder.half()
else:
    print("Using CPU")

# 3. Generate frames with motion progression
print("\nGenerating animation frames...")
num_frames = 16  # Reduced for testing
height, width = 512, 512
frames = []

motion_phases = [
    (0.0, 0.3, "raising hands to begin clapping"),
    (0.3, 0.6, "bringing hands together to clap"),
    (0.6, 0.9, "hands clapping together"),
    (0.9, 1.0, "lowering hands after clapping")
]

for i in range(num_frames):
    progress = i / num_frames
    
    # Find current motion phase
    current_action = next(
        (action for start, end, action in motion_phases if start <= progress < end),
        "standing still"
    )
    
    prompt = f"a person {current_action}, highly detailed, 4k resolution, professional photography"
    negative_prompt = "blurry, deformed, low quality, bad anatomy, extra limbs"
    
    # Generate frame
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=25,  # Reduced for testing
            guidance_scale=7.5
        ).images[0]
    
    frames.append(np.array(image))
    print(f"Generated frame {i+1}/{num_frames} - {current_action}")

# 4. Create MP4 video
print("\nCreating video...")
video_path = os.path.join(output_dir, "clapping_animation.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, 8.0, (width, height))  # Lower FPS for testing

for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
print(f"\nVideo successfully created at: {video_path}")
