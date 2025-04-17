from flask import Flask, request, send_file, render_template,send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip
import os
import numpy as np 

app = Flask(_name_, static_url_path='/static')
CORS(app)

# Load the pipeline once
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/generate-video", methods=["POST"])
def generate_video():
    image_file = request.files["image"]
    prompt = request.form["prompt"]
    
    # Open and process the image
    img = Image.open(image_file).convert("RGB")
    img = img.resize((384, 384))

    # Convert image to tensor format
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Generate video frames based on image and prompt
    video_frames = pipe(prompt=prompt, init_image=img_tensor, num_frames=6).frames[0]

    # Ensure video_frames is a list of images
    if isinstance(video_frames, Image.Image):
        video_frames = [video_frames]

    # Save video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        clip = ImageSequenceClip([np.array(frame) for frame in video_frames], fps=7)
        clip.write_videofile(temp.name, codec="libx264", audio=False)
        return send_file(temp.name, mimetype="video/mp4", as_attachment=True, download_name="output.mp4")

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=8080)
