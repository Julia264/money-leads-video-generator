import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed
import logging

# 📋 Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🟡 Dataset with improved error handling
class FrameDataset(Dataset):
    def __init__(self, root_dir, prompt_dict, image_size=512):
        self.samples = []
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                frames = [os.path.join(label_path, f) for f in os.listdir(label_path) 
                         if f.endswith((".png", ".jpg", ".jpeg"))]
                for f in frames:
                    self.samples.append((f, prompt_dict.get(label, "a person")))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, prompt = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            # Validate image tensor
            if torch.isnan(image).any() or torch.isinf(image).any():
                logger.warning(f"Invalid pixel values in {path}")
                return None, None
            return image, prompt
        except Exception as e:
            logger.warning(f"Error loading image {path}: {str(e)}")
            return None, None

# ⚙️ Safe collate_fn
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    images, prompts = zip(*batch)
    images = torch.stack(images)
    # Double-check for NaN/Inf before returning
    if torch.isnan(images).any() or torch.isinf(images).any():
        return None, None
    return images, list(prompts)

# 🛠️ LoRA Injection with lower rank
def inject_lora(unet, r=2):  # Reduced rank for stability
    inject_trainable_lora(
        unet,
        r=r,
        target_replace_module=["CrossAttention", "Attention"],
    )

# 🔵 Training with NaN protection
def train_lora(data_dir, prompts, output_dir):
    # Initialize accelerator with gradient accumulation
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision='fp16'
    )
    set_seed(42)

    # Enable memory efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Load pipeline with more stable settings
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(accelerator.device)
    
    # Use DDPM scheduler for more stable training
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze all components except LoRA
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Load text components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
    text_encoder = text_encoder.to(dtype=torch.float16)

    # Inject LoRA with lower rank
    inject_lora(pipe.unet)

    # Prepare dataset with validation
    dataset = FrameDataset(data_dir, prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    # Conservative optimizer settings
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=1e-5,  # Reduced learning rate
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(dataloader)*3,
        eta_min=1e-6
    )

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Prepare with accelerator
    pipe.unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, dataloader, lr_scheduler
    )

    # Training loop with NaN checks
    pipe.unet.train()
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            if batch is None or batch[0] is None:
                continue

            images, captions = batch
            
            with accelerator.accumulate(pipe.unet):
                images = images.to(accelerator.device, dtype=torch.float16)

                # Encode images to latents with validation
                with torch.no_grad():
                    latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                    if torch.isnan(latents).any():
                        logger.warning("NaN detected in latents, skipping batch")
                        continue

                # Encode text
                input_ids = tokenizer(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids).last_hidden_state
                    if torch.isnan(encoder_hidden_states).any():
                        logger.warning("NaN detected in text embeddings, skipping batch")
                        continue

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), 
                    device=latents.device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Mixed precision training with gradient scaling
                with accelerator.autocast():
                    model_pred = pipe.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample

                    # Calculate loss in float32 for stability
                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(), 
                        noise.float(), 
                        reduction="mean"
                    )

                # Check for NaN before backprop
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"NaN/Inf loss detected at step {step}, skipping")
                    optimizer.zero_grad()
                    continue

                # Backprop with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Logging
            if step % 10 == 0:
                lr = lr_scheduler.get_last_lr()[0]
                accelerator.print(
                    f"Epoch {epoch+1} Step {step}: "
                    f"Loss = {loss.item():.4f}, "
                    f"LR = {lr:.2e}"
                )

    # Save model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipe.save_pretrained(output_dir)
        logger.info(f"✅ Training complete! Model saved at: {output_dir}")

# 🚀 Main
if __name__ == "__main__":
    BASE_DIR = os.getcwd()

    prompts = {
        "احبك": "a person saying I love you",
        "احسنت": "a person saying Well done happily",
        "اعجبني": "a person showing approval",
        "انت عظيم": "a person saying You are great",
        "تصفيق": "a person clapping",
        "حبيبي": "a person saying my dear warmly",
        "مرحبا": "a person waving hello",
        "هذا رائع": "a person saying This is wonderful",
        "واو": "a person saying Wow!",
        "مدهش": "a person looking amazed"
    }

    data_dir = os.path.join(BASE_DIR, "datasets", "frames")
    output_dir = os.path.join(BASE_DIR, "models", "fine-tuned-motion")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    os.makedirs(output_dir, exist_ok=True)

    train_lora(data_dir, prompts, output_dir)
