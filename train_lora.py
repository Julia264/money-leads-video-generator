import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.training_utils import set_seed
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🟢 Dataset class for PNG frames
class FrameDataset(Dataset):
    def __init__(self, root_dir, prompt_dict, image_size=512):
        self.samples = []
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for f in os.listdir(label_path):
                    if f.endswith(".png"):
                        self.samples.append((os.path.join(label_path, f), prompt_dict.get(label, "a person")))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, prompt = self.samples[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, prompt

# 🟡 Inject LoRA
def inject_lora(unet):
    unet.set_attn_processor(LoRAAttnProcessor())

# 🔵 Main training function
def train_lora(data_dir, prompts, output_dir):
    accelerator = Accelerator()
    set_seed(42)

    logger.info("Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(accelerator.device)

    # Inject LoRA into UNet
    inject_lora(pipe.unet)
    logger.info("LoRA layers injected.")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)

    # Load dataset
    dataset = FrameDataset(data_dir, prompts)
    if len(dataset) == 0:
        logger.info("Dataset loaded with 0 samples")
        raise ValueError("No valid samples found!")

    logger.info(f"Dataset loaded with {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)
    pipe.unet.train()

    for epoch in range(3):
        total_loss = 0
        for step, (images, captions) in enumerate(dataloader):
            with accelerator.accumulate(pipe.unet):
                images = images.to(accelerator.device, dtype=torch.float16)
                input_ids = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

                noise = torch.randn_like(images)
                timesteps = torch.randint(0, 1000, (images.shape[0],), device=images.device).long()
                noisy_images = pipe.scheduler.add_noise(images, noise, timesteps)

                model_pred = pipe.unet(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"⚠️ Invalid loss detected at step {step}, skipping...")
                    continue

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                if step % 10 == 0:
                    logger.info(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")

        logger.info(f"Epoch {epoch+1} Avg Loss = {total_loss / len(dataloader):.4f}")

    accelerator.wait_for_everyone()
    pipe.save_pretrained(output_dir)
    logger.info("✅ Training complete. Model saved.")

# 🔻 Run if main
if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    prompts = {
        "clapping": "a person clapping hands",
        "waving": "a person waving hello",
        "مرحبا": "a person saying hello",
        "تصفيق": "a person applauding"
    }

    data_dir = os.path.join(BASE_DIR, "Dataset2", "train")
    output_dir = os.path.join(BASE_DIR, "lora_model")

    os.makedirs(output_dir, exist_ok=True)
    train_lora(data_dir, prompts, output_dir)
