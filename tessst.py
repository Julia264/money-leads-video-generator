import torch
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image

# 1. Setup Pipeline with Your LoRA
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)

# Load your trained LoRA weights
lora_path = "/home/ubuntu/money-leads-video-generator/lora_model/best_model"
pipe.load_lora_weights(lora_path, adapter_name="clapping")

# 2. Generate Frames with Progressive Prompts
num_frames = 24  # 2 seconds at 12fps
frames = []

print("Generating animation frames...")
for i in range(num_frames):
    # Progressively change the prompt through the clapping motion
    progress = i / num_frames
    if progress < 0.3:
        action = "starting to bring hands together"
    elif progress < 0.7:
        action = "clapping hands together"
    else:
        action = "bringing hands apart after clapping"
    
    prompt = f"a person {action}, high quality, detailed, 4k"
    
    # Generate frame
    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, deformed, low quality",
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]
    
    frames.append(np.array(image))
    print(f"Generated frame {i+1}/{num_frames}")

# 3. Create MP4 Video
print("Creating MP4 video...")
height, width = frames[0].shape[:2]
video = cv2.VideoWriter(
    'clapping_animation.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'),
    12,  # FPS
    (width, height)
)

for frame in frames:
    # Convert RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(bgr_frame)

video.release()
print("Video saved as clapping_animation.mp4")
