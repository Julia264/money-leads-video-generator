from flask import Flask, request, send_file, render_template,send_from_directory
from flask_cors import CORS
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
import torch
import tempfile
from moviepy import ImageSequenceClip
import numpy as np 
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
app = Flask(__name__, static_url_path='/static')
CORS(app)

# Load the pipeline once
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    #torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to(pipe.to("cpu")
       )


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/generate-video", methods=["POST"])
def generate_video():
    image_file = request.files["image"]
    prompt = request.form["prompt"]

    # معالجة الصورة
    img = Image.open(image_file).convert("RGB")
    img = img.resize((384, 384))

    
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()

  
    video_frames = pipe(
        prompt=prompt,
        image=img_tensor,
        num_frames=4,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
    ).frames[0]

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        clip = ImageSequenceClip(video_frames, fps=7)
        clip.write_videofile(temp.name, codec="libx264", audio=False)
        return send_file(temp.name, mimetype="video/mp4", as_attachment=True, download_name="output.mp4")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
