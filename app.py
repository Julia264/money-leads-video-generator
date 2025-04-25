# app.py
from flask import Flask, request, send_file
from flask_cors import CORS
from diffusers import AnimateDiffPipeline
from PIL import Image
import torch
import numpy as np
from moviepy import ImageSequenceClip
import tempfile

app = Flask(__name__)
CORS(app)

pipe = AnimateDiffPipeline.from_pretrained(
    "./models/fine-tuned-motion",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_MAP = {
    "clap": "a person clapping hands",
    "wave": "a person waving",
    "thumbs_up": "a person giving thumbs up"
}

@app.route("/generate", methods=["POST"])
def generate():
    image = Image.open(request.files["image"]).convert("RGB")
    action = request.form.get("action", "clap")
    prompt = PROMPT_MAP.get(action, "a person clapping hands")

    image = image.resize((512, 512))
    frames = pipe(prompt=prompt, image=image, num_frames=16).frames[0]
    frames = [np.array(f) for f in frames]

    clip = ImageSequenceClip(frames, fps=10)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    clip.write_videofile(temp.name, codec="libx264", audio=False)

    return send_file(temp.name, as_attachment=True, download_name="generated.mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
