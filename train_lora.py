import os
import torch
import zipfile
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed

# 🛠 Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🟡 Dataset
class TwoActionDataset(Dataset):
    def __init__(self, zip_path, image_size=512):
        self.samples = []
        self.zip_path = zip_path
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
        ])

        with zipfile.ZipFile(zip_path, 'r') as z:
            for file in z.namelist():
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = None
                    if 'clapping' in file.lower():
                        label = "a person clapping hands"
                    elif 'waving' in file.lower():
                        label = "a person waving hello"
                    if label:
                        self.samples.append((file, label))

        logger.info(f"Dataset loaded with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, prompt = self.samples[idx]
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as z:
                with z.open(file_name) as f:
                    image = Image.open(f).convert("RGB")
                    tensor = self.transform(image)
                    tensor = tensor.clamp(-1, 1)

                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        return None, None

                    return tensor.half(), prompt
        except Exception as e:
            logger.warning(f"Failed to load {file_name}: {e}")
            return None, None

def safe_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    images, prompts = zip(*batch)
    return torch.stack(images), list(prompts)

# 🟢 LoRA Injection
def inject_lora_layers(unet, r=4):
    inject_trainable_lora(unet, r=r, target_replace_module=["CrossAttention", "Attention"])
    logger.info("LoRA layers injected.")

# 🔵 Train Function
def train_lora(zip_path, output_dir):
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
    set_seed(42)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(accelerator.device)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).half()

    inject_lora_layers(pipe.unet)

    dataset = TwoActionDataset(zip_path)
    if len(dataset) == 0:
        raise ValueError("No valid samples found!")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    logger.info(f"TRAIN set: {len(train_dataset)} valid samples")
    logger.info(f"VAL set: {len(val_dataset)} valid samples")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=safe_collate)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*5)

    pipe.unet, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_loader, val_loader, lr_scheduler
    )

    num_epochs = 5
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting training...")

    for epoch in range(num_epochs):
        pipe.unet.train()
        total_loss = 0
        valid_steps = 0

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue
            images, captions = batch
            images = images.to(accelerator.device, dtype=torch.float16)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            with accelerator.autocast():
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            total_loss += loss.item()
            valid_steps += 1

            if step % 10 == 0:
                avg_loss = total_loss / (valid_steps or 1)
                logger.info(f"Epoch {epoch+1} Step {step}: Loss = {avg_loss:.4f}")

        avg_epoch_loss = total_loss / (valid_steps or 1)
        logger.info(f"Epoch {epoch+1} Avg Loss = {avg_epoch_loss:.4f}")

        # Validation
        pipe.unet.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, captions = batch
                images = images.to(accelerator.device, dtype=torch.float16)
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / (val_steps or 1)
        logger.info(f"Epoch {epoch+1} Validation Loss = {avg_val_loss:.4f}")

    # Final save
    if accelerator.is_main_process:
        pipe.save_pretrained(os.path.join(output_dir, "final_model"))
        logger.info("Training complete. Final model saved.")

if __name__ == "__main__":
    zip_path = "/home/ubuntu/money-leads-video-generator/Dataset2.zip"
    output_dir = "/home/ubuntu/money-leads-video-generator/models"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip file not found at {zip_path}")

    train_lora(zip_path, output_dir)
