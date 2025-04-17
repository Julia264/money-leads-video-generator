from flask import Flask, request, send_file, render_template
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip
import os

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
    image_file = request.files["image"]
    img = Image.open(image_file).convert("RGB")
    img = img.resize((384, 384))

    video_frames = pipe(img, num_frames=6).frames[0]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        clip = ImageSequenceClip(video_frames, fps=7)
        clip.write_videofile(temp.name, codec="libx264", audio=False, verbose=False, logger=None)
        return send_file(temp.name, mimetype="video/mp4", as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
