import os
import torch
import zipfile
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PTFileDataset(Dataset):
    def __init__(self, zip_path, split='train', action='clapping'):
        self.zip_path = zip_path
        self.split = split
        self.action = action
        self.file_paths = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if (f"{split}/{action}/" in file and 
                    file.endswith('.pt') and 
                    not file.startswith('__MACOSX')):
                    self.file_paths.append(file)
        
        logger.info(f"Found {len(self.file_paths)} .pt files for {split}/{action}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(file_path) as pt_file:
                    tensor = torch.load(pt_file, map_location='cpu')
                    
                    # Convert to FP32 first if needed
                    if tensor.dtype == torch.float16:
                        tensor = tensor.float()
                    
                    # Validate tensor
                    if tensor.dim() != 3 or tensor.size(0) != 3:
                        logger.warning(f"Unexpected tensor shape {tensor.shape} in {file_path}")
                        return None, None
                        
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        logger.warning(f"Invalid tensor values in {file_path}")
                        return None, None
                        
                    # Normalize images to [-1, 1] if they aren't already
                    if tensor.min() >= 0 and tensor.max() > 1:
                        tensor = (tensor / 127.5) - 1.0
                        
                    prompt = {
                        'clapping': 'a person clapping hands',
                        'waving': 'a person waving hello'
                    }.get(self.action, f"a person {self.action}")
                    
                    return tensor, prompt
                    
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {str(e)}")
            return None, None

def safe_collate(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
        
    images, prompts = zip(*batch)
    return torch.stack(images), list(prompts)

def inject_lora(unet, r=4):
    try:
        inject_trainable_lora(
            unet,
            r=r,
            target_replace_module=["CrossAttention", "Attention"],
        )
        logger.info("Successfully injected LoRA layers")
    except Exception as e:
        logger.error(f"Failed to inject LoRA: {str(e)}")
        raise

def train_lora(zip_path, output_dir, action='clapping'):
    # Initialize accelerator with mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16'
    )
    set_seed(42)
    
    # Enable memory efficient attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Load pipeline with FP16 weights
    logger.info("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(accelerator.device)
    
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze components
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Load text components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
    text_encoder = text_encoder.to(dtype=torch.float16)

    # Inject LoRA
    inject_lora(pipe.unet)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = PTFileDataset(zip_path, split='train', action=action)
    val_dataset = PTFileDataset(zip_path, split='val', action=action)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=safe_collate,
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=safe_collate,
        persistent_workers=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_dataloader) * 5,
        eta_min=1e-6
    )

    # Prepare with accelerator
    pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Training loop
    num_epochs = 5
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        pipe.unet.train()
        epoch_loss = 0
        valid_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None or batch[0] is None:
                continue

            images, captions = batch
            images = images.to(accelerator.device)
            
            # Convert images to FP32 for VAE encoding
            images = images.float()

            with accelerator.autocast():
                with torch.no_grad():
                    # Encode images to latents
                    latents = pipe.vae.encode(images).latent_dist.sample()
                    latents = latents * 0.18215  # Scale factor
                    
                    # Encode text
                    input_ids = tokenizer(
                        captions,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).input_ids.to(accelerator.device)
                    encoder_hidden_states = text_encoder(input_ids).last_hidden_state

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), 
                    device=latents.device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Forward pass
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), 
                    noise.float(), 
                    reduction="mean"
                )

            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            
            if step % 10 == 0:
                avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step}/{len(train_dataloader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                )

        # Validation
        pipe.unet.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None or batch[0] is None:
                    continue
                    
                images, captions = batch
                images = images.to(accelerator.device).float()
                
                with accelerator.autocast():
                    latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                    input_ids = tokenizer(
                        captions,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).input_ids.to(accelerator.device)
                    encoder_hidden_states = text_encoder(input_ids).last_hidden_state
                    
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, pipe.scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=latents.device
                    ).long()
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    
                    model_pred = pipe.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
                    val_loss += loss.item()
                    val_batches += 1

        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            logger.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if accelerator.is_main_process:
                    save_path = os.path.join(output_dir, "best_model")
                    pipe.save_pretrained(save_path, safe_serialization=True)
                    logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = os.path.join(output_dir, "final_model")
        pipe.save_pretrained(final_save_path, safe_serialization=True)
        logger.info(f"Training complete! Model saved at: {final_save_path}")

if __name__ == "__main__":
    zip_path = "/home/ubuntu/money-leads-video-generator/Dataset2.zip"
    output_dir = "/home/ubuntu/money-leads-video-generator/lora_model"
    action = "clapping"  # Change to "waving" if needed
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip file not found at {zip_path}")
    
    try:
        train_lora(zip_path, output_dir, action)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
