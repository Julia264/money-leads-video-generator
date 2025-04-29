
import os
import torch
import zipfile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoActionDataset(Dataset):
    def __init__(self, zip_path, image_size=512):
        self.samples = []
        self.zip_path = zip_path
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if 'clapping' in file_name.lower() and file_name.endswith('.pt'):
                    self.samples.append((file_name, "a person clapping hands"))
                elif 'waving' in file_name.lower() and file_name.endswith('.pt'):
                    self.samples.append((file_name, "a person waving hello"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, prompt = self.samples[idx]
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(file_name) as pt_data:
                    tensor = torch.load(pt_data, map_location="cpu")
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
                    elif tensor.shape[0] != 3:
                        return None, None
                    if torch.all(tensor == 0) or torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        return None, None
                    tensor = (tensor / 255.0) * 2.0 - 1.0
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                    return tensor.float(), prompt
        except Exception:
            return None, None

def safe_collate(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    images, prompts = zip(*batch)
    return torch.stack(images), list(prompts)

def inject_lora(unet, r=4):
    inject_trainable_lora(unet, r=r, target_replace_module=["CrossAttention", "Attention"])

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

    inject_lora(pipe.unet)

    dataset = TwoActionDataset(zip_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=safe_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=safe_collate)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*5, eta_min=1e-6)

    pipe.unet, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    for epoch in range(5):
        pipe.unet.train()
        for step, batch in enumerate(train_dataloader):
            if batch is None or batch[0] is None:
                continue
            images, captions = batch
            images = images.to(accelerator.device)

            with torch.no_grad():
                latents = pipe.vae.encode(images.float()).latent_dist.sample().float() * 0.18215
                input_ids = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state.float()

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            with accelerator.autocast():
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                optimizer.zero_grad()
                continue

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")

    if accelerator.is_main_process:
        pipe.save_pretrained(os.path.join(output_dir, "final_model"))
        print("✅ Training complete! Model saved.")

if __name__ == "__main__":
    zip_path = "/home/ubuntu/money-leads-video-generator/Dataset2.zip"
    output_dir = "/home/ubuntu/money-leads-video-generator/lora_model"
    os.makedirs(output_dir, exist_ok=True)
    train_lora(zip_path, output_dir)
