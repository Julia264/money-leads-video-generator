
# Money Leads Video Generator 🎥✨

## 📂 Folder Structure

```
money-leads-video-generator/
├── app.py                # Flask web app for generating animated videos
├── extract_frames.py      # Script to extract frames from videos into folders
├── train_lora.py          # Script to train AnimateDiff model with LoRA fine-tuning
├── datasets/              # Folder containing videos (and extracted frames)
│   └── الحركات/             # Arabic-labeled folders with extracted frames
├── models/                # Folder to save pre-trained models and fine-tuned models
│   └── mm_sd_v14.ckpt      # Base model checkpoint
│   └── mm_sd_v15.ckpt      # Motion module checkpoint
├── static/                # Folder for frontend files (HTML, CSS if needed)
```

---

## ⚙️ How It Works

- **Extract Frames:**  
  - Run `extract_frames.py` to split each video into individual frame images.

- **Train LoRA Model:**  
  - Run `train_lora.py` to fine-tune the AnimateDiff model using the extracted frames.

- **Generate Videos:**  
  - Use `app.py` to run a Flask web server where you upload an image and choose an action (like clap, wave, thumbs up) to generate a video.

---

## 🛠️ Requirements

Install the needed packages:

```bash
pip install -r requirements.txt
```

If you face tokenizers issues, run:

```bash
pip install --upgrade pip
pip install tokenizers --prefer-binary
```

---

## 🚀 Full Steps to Run

1. **Clone the repository and set up environment**

```bash
git clone https://github.com/Julia264/money-leads-video-generator.git
cd money-leads-video-generator
python3 -m venv lora-env
source lora-env/bin/activate
pip install -r requirements.txt
```

2. **Upload Videos**
   - Place your `.mp4` or `.mov` videos inside `datasets/الحركات/`.

3. **Extract Frames**

```bash
python3 extract_frames.py
```

4. **Download and Prepare Models**
   - Download `mm_sd_v14.ckpt` and `mm_sd_v15.ckpt` and put them inside the `models/` folder.

5. **Train the LoRA model**

```bash
python3 train_lora.py
```

6. **Run the Web Application**

```bash
python3 app.py
```
- Access the app from your browser at: `http://localhost:8080`

---

## 📌 Notes

- **Frames Directory:** Make sure the `datasets/الحركات/` folder contains extracted frames before training.
- **GPU Recommended:** Training and generation require a CUDA-compatible GPU for speed.
- **Authentication:** HuggingFace login may be needed to download models.

---

## 👩‍💻 Author

Developed by [Julia264](https://github.com/Julia264).
