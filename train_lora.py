import os
import torch
import zipfile
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PTActionDataset(Dataset):
    def __init__(self, zip_path, split="train"):
        self.samples = []
        self.zip_path = zip_path

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if split in file_name and file_name.endswith(".pt"):
                    if "clapping" in file_name.lower():
                        self.samples.append((file_name, "a person clapping hands"))
                    elif "waving" in file_name.lower():
                        self.samples.append((file_name, "a person waving hello"))

        logger.info(f"{split.upper()} set: {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, prompt = self.samples[idx]
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(file_name) as pt_file:
                    tensor = torch.load(pt_file, map_location="cpu")

                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
                    elif tensor.shape[0] != 3:
                        return None, None

                    if torch.isnan(tensor).any() or torch.isinf(tensor).any() or torch.all(tensor == 0):
                        return None, None

                    tensor = (tensor / 255.0) * 2.0 - 1.0
                    tensor = torch.clamp(tensor, -1.0, 1.0)

                    return tensor.half(), prompt
        except Exception as e:
            logger.warning(f"Failed loading {file_name}: {str(e)}")
            return None, None

def safe_collate(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    images, prompts = zip(*batch)
    return torch.stack(images), list(prompts)

def inject_lora_layers(unet):
    inject_trainable_lora(unet, r=4, target_replace_module=["CrossAttention", "Attention"])
    logger.info("LoRA layers injected.")

def train_lora(zip_path, output_dir):
    accelerator = Accelerator()
    set_seed(42)

    logger.info("Loading pipeline...")
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
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)

    inject_lora_layers(pipe.unet)

    train_dataset = PTActionDataset(zip_path, split="train")
    val_dataset = PTActionDataset(zip_path, split="val")

    if len(train_dataset) == 0:
        raise ValueError("No training data found.")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=safe_collate)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-5)
    pipe.unet, optimizer, train_loader, val_loader = accelerator.prepare(pipe.unet, optimizer, train_loader, val_loader)

    logger.info("Starting training...")
    best_loss = float("inf")
    for epoch in range(5):
        pipe.unet.train()
        total_loss, valid_batches = 0, 0
        for step, batch in enumerate(train_loader):
            if batch is None: continue
            images, prompts = batch
            images = images.to(accelerator.device)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state.half()

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            with accelerator.autocast():
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            valid_batches += 1

            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / valid_batches if valid_batches else float("inf")
        logger.info(f"Epoch {epoch+1} Avg Loss = {avg_loss:.4f}")

        # Evaluate
        pipe.unet.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                images, prompts = batch
                images = images.to(accelerator.device)
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state.half()
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                val_loss += torch.nn.functional.mse_loss(model_pred.float(), noise.float()).item()

        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1} Validation Loss = {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            if accelerator.is_main_process:
                pipe.save_pretrained(os.path.join(output_dir, "best_model"))
                logger.info("Saved best model")

    if accelerator.is_main_process:
        pipe.save_pretrained(os.path.join(output_dir, "final_model"))
        logger.info("Training complete. Final model saved.")

if __name__ == "__main__":
    zip_path = "/home/ubuntu/money-leads-video-generator/Dataset2.zip"
    output_dir = "/home/ubuntu/money-leads-video-generator/lora_model"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")
    train_lora(zip_path, output_dir)
