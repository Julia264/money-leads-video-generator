from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip,ImageClip
from moviepy.video.fx import resize
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
    img = img.resize((224, 224))  # Resize to fit the model input size (if needed)

    # Convert image to tensor and normalize it
    img_array = np.array(img) / 255.0  # Normalize the image
    img_tensor = torch.tensor(img_array).unsqueeze(0).permute(0, 3, 1, 2)  # Add batch dimension and permute to [B, C, H, W]
    
    # Ensure the tensor is of type float32
    img_tensor = img_tensor.to(torch.float32)
    
    # Generate video from the image (num_frames is set to 6 in this example)
    video_frames = pipe(img_tensor, num_frames=6).frames[0]

    # Convert each frame to a numpy array if it's a PIL image
    video_frames = [np.array(frame) for frame in video_frames]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        clip = ImageSequenceClip(video_frames, fps=500)
        # Set the video duration to 5 seconds (for 30fps, 5 seconds = 5 * 30 = 150 frames)
        clip = clip.with_duration(5).resize(width=1920)  # Set video duration to 5 seconds
        # Apply a zoom-in effect (lambda t: 1 + 0.1 * t creates a zoom-in over time)
        zoomed_clip = image_clip.resize(lambda t: 1 + 0.1 * t)
        zoomed_clip = zoomed_clip.fadein(1)  # Fade in over 1 second
        clip.write_videofile(temp.name, codec="libx264", audio=False)

        # Send the generated video back to the client
        return send_file(temp.name, mimetype="video/mp4", as_attachment=True, download_name="output.mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
