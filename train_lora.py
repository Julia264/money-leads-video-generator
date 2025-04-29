import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Dataset
class ClappingTensorDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for split in ['train', 'val']:
            split_dir = os.path.join(root_dir, split, 'clapping')
            if not os.path.exists(split_dir):
                continue
            for fname in os.listdir(split_dir):
                if fname.endswith('.pt'):
                    self.samples.append((
                        os.path.join(split_dir, fname),
                        "a person clapping hands",
                        split
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, prompt, split = self.samples[idx]
        try:
            tensor = torch.load(path, map_location="cpu")
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            elif tensor.shape[0] != 3:
                return None, None, None
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return None, None, None
            tensor = (tensor / 255.0) * 2.0 - 1.0
            tensor = torch.clamp(tensor, -1.0, 1.0)
            return tensor.half(), prompt, split
        except:
            return None, None, None

def safe_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    imgs, prompts, _ = zip(*batch)
    return torch.stack(imgs), list(prompts)

# ✅ LoRA
def inject_lora_layers(unet):
    inject_trainable_lora(
        unet,
        r=4,
        target_replace_module=["CrossAttention", "Attention"],
    )
    logger.info("LoRA layers injected.")

# ✅ Train
def train_lora(data_dir, output_dir):
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

    dataset = ClappingTensorDataset(data_dir)
    if len(dataset) == 0:
        raise ValueError("No valid samples found!")

    train_data = [d for d in dataset if d[2] == "train"]
    val_data = [d for d in dataset if d[2] == "val"]

    logger.info(f"TRAIN set: {len(train_data)} valid samples")
    logger.info(f"VAL set: {len(val_data)} valid samples")

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=safe_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=safe_collate)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-5)
    pipe.unet, optimizer, train_loader, val_loader = accelerator.prepare(pipe.unet, optimizer, train_loader, val_loader)

    for epoch in range(5):
        pipe.unet.train()
        running_loss = 0
        for step, batch in enumerate(train_loader):
            if batch is None:
                continue
            images, prompts = batch
            images = images.to(accelerator.device)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

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
            running_loss += loss.item()

            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")

        logger.info(f"Epoch {epoch+1} Avg Loss = {running_loss / (step + 1):.4f}")

        # Validation
        pipe.unet.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, prompts = batch
                images = images.to(accelerator.device)
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                input_ids = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                val_loss += torch.nn.functional.mse_loss(model_pred.float(), noise.float()).item()

        logger.info(f"Epoch {epoch+1} Validation Loss = {val_loss / len(val_loader):.4f}")

    pipe.save_pretrained(output_dir)
    logger.info("Training complete. Final model saved.")

# ✅ Run
if __name__ == "__main__":
    data_dir = "/home/ubuntu/money-leads-video-generator/Dataset2"
    output_dir = "/home/ubuntu/money-leads-video-generator/lora_model"
    os.makedirs(output_dir, exist_ok=True)
    train_lora(data_dir, output_dir)
