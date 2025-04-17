from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip
from moviepy.editor import VideoFileClip
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

    # Generate video from the image (num_frames is set to 6 in this example)
    video_frames = pipe(img_tensor, num_frames=6).frames[0]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        clip = ImageSequenceClip(video_frames, fps=7)
        temp_path = temp.name
        clip.write_videofile(temp_path, codec="libx264", audio=False)

        # Post-processing: Upscale video to 1920x1080 using FFmpeg
        output_path = "/path/to/your/output/video/output_upscaled.mp4"
        ffmpeg_command = [
            "ffmpeg", "-i", temp_path, "-vf", "scale=1920:1080", "-c:a", "aac", output_path
        ]
        subprocess.run(ffmpeg_command, check=True)

        # Load the generated video file
        video_clip = VideoFileClip(output_path)

        # Add motion to the video by resizing and changing the duration
        # Set the duration of the video to 5 seconds
        video_clip = video_clip.set_duration(5)

        # Optionally, apply some motion effect (for example: a zoom-in effect)
        video_clip = video_clip.resize(lambda t: 1 + 0.1 * t)  # Scaling effect (zoom-in over time)

        # Optionally, apply a fade-in effect to make it smoother
        video_clip = video_clip.fadein(1)  # 1 second fade-in effect

        # Write the final video to a file with motion effect
        final_output_path = "/path/to/your/output/video/final_video_with_motion.mp4"
        video_clip.write_videofile(final_output_path, codec="libx264", audio=False)

        # Send the generated video back to the client
        return send_file(final_output_path, mimetype="video/mp4", as_attachment=True, download_name="final_video_with_motion.mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
