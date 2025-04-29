import os
import torch
import zipfile
import numpy as np
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Dataset --------------------
class ClappingDataset(Dataset):
    def __init__(self, zip_path, subset, image_size=512):
        self.samples = []
        self.zip_path = zip_path
        self.subset = subset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            self.samples = [
                (name, "a person clapping hands")
                for name in zip_ref.namelist()
                if name.startswith(f'Dataset2/{subset}/clapping') and name.endswith('.pt')
            ]
        
        logger.info(f"{subset.upper()} set: {len(self.samples)} valid samples")

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
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        return None, None
                    tensor = torch.clamp(tensor, -1.0, 1.0)
                    return tensor.half(), prompt
        except Exception as e:
            logger.warning(f"Failed to load {file_name}: {str(e)}")
            return None, None

def safe_collate(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    images, prompts = zip(*batch)
    return torch.stack(images), list(prompts)

# -------------------- LoRA Training --------------------
def inject_lora_layers(unet, r=4):
    inject_trainable_lora(unet, r=r, target_replace_module=["CrossAttention", "Attention"])
    logger.info("LoRA layers injected.")

def train_lora(zip_path, output_dir):
    accelerator = Accelerator()
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
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
    text_encoder = text_encoder.to(dtype=torch.float16)

    inject_lora_layers(pipe.unet)

    train_dataset = ClappingDataset(zip_path, subset="train")
    val_dataset = ClappingDataset(zip_path, subset="val")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=safe_collate)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)

    pipe.unet, optimizer, train_loader, val_loader = accelerator.prepare(pipe.unet, optimizer, train_loader, val_loader)

    best_loss = float("inf")
    for epoch in range(5):
        pipe.unet.train()
        total_loss, steps = 0, 0
        for step, batch in enumerate(train_loader):
            if batch is None or batch[0] is None:
                continue
            images, captions = batch
            images = images.to(accelerator.device)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            with accelerator.autocast():
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1

            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / steps if steps > 0 else float("inf")
        logger.info(f"Epoch {epoch+1} Avg Loss = {avg_loss:.4f}")

        # Validation
        pipe.unet.eval()
        val_loss, val_steps = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch[0] is None:
                    continue
                images, captions = batch
                images = images.to(accelerator.device)
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else float("nan")
        logger.info(f"Epoch {epoch+1} Validation Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            pipe.save_pretrained(os.path.join(output_dir, "best_model"))
    
    pipe.save_pretrained(os.path.join(output_dir, "final_model"))
    logger.info("Training complete. Final model saved.")

# -------------------- Entry --------------------
if __name__ == "__main__":
    zip_path = "/home/ubuntu/money-leads-video-generator/Dataset2.zip"
    output_dir = "/home/ubuntu/money-leads-video-generator/lora_model"
    os.makedirs(output_dir, exist_ok=True)
    train_lora(zip_path, output_dir)
