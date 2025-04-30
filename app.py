from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip
import numpy as np

import torch
torch.cuda.empty_cache()

app = Flask(__name__, static_url_path='/static')
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
    # Get the uploaded image
    image_file = request.files["image"]
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))  # Resize to fit the model input size

    # Convert image to tensor, normalize, and cast to float32
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)

    # Generate video
    video_frames = pipe(img_tensor, num_frames=6).frames[0]

    # Convert PIL images to numpy arrays for MoviePy
    video_frames_np = [np.array(frame) for frame in video_frames]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        clip = ImageSequenceClip(video_frames_np, fps=7)
        clip.write_videofile(temp.name, codec="libx264", audio=False)

        return send_file(temp.name, mimetype="video/mp4", as_attachment=True, download_name="output.mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
