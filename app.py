from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip
import numpy as np

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
    img = img.resize((224, 224), Image.LANCZOS)

    # Convert image to tensor and normalize it
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).permute(0, 3, 1, 2)
    img_tensor = img_tensor.to(torch.float32)

    # Generate video frames
    video_frames = pipe(img_tensor, num_frames=6).frames[0]
    video_frames = [np.array(frame) for frame in video_frames]

    # Create video clip
    clip = ImageSequenceClip(video_frames, fps=30)
    clip = clip.with_duration(5)

    # Export video in high quality
    # Export video in high quality
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        #clip = clip.resize(height=1080, width=1920)  # upscale after generation
        clip.write_videofile(
            temp.name,
            codec="libx264",
            bitrate="10000k",
            fps=30,
            preset="slow",
            audio=False
        )

        return send_file(
            temp.name,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="output_with_motion.mp4"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
