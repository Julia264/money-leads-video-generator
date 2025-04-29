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
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustTwoActionDataset(Dataset):
    def __init__(self, zip_path, image_size=512):
        self.samples = []
        self.zip_path = zip_path
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Track valid/invalid files
        self.valid_files = 0
        self.invalid_files = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    action = None
                    if 'clapping' in file_name.lower():
                        action = ('clapping', "a person clapping hands")
                    elif 'waving' in file_name.lower():
                        action = ('waving', "a person waving hello")
                    
                    if action:
                        self.samples.append((file_name, action[1]))
                        self.valid_files += 1
                    else:
                        self.invalid_files += 1
        
        logger.info(f"Found {self.valid_files} valid images and {self.invalid_files} invalid/unrecognized images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, prompt = self.samples[idx]
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(file_name) as img_file:
                    img = Image.open(img_file).convert('RGB')
                    
                    # Apply transformations
                    img_tensor = self.transform(img)
                    
                    # Validate tensor
                    if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                        logger.debug(f"Invalid tensor values in {file_name}")
                        return None, None
                        
                    if img_tensor.min() < -1 or img_tensor.max() > 1:
                        logger.debug(f"Tensor values out of range in {file_name}")
                        return None, None
                        
                    return img_tensor, prompt
        except Exception as e:
            logger.debug(f"Error loading {file_name}: {str(e)}")
            return None, None

def safe_collate(batch):
    """Enhanced collate function with better error handling"""
    batch = [item for item in batch if item[0] is not None]
    
    if len(batch) == 0:
        logger.warning("Empty batch detected - skipping")
        return None, None
        
    images, prompts = zip(*batch)
    
    try:
        images_tensor = torch.stack(images)
        
        # Additional validation
        if torch.isnan(images_tensor).any() or torch.isinf(images_tensor).any():
            logger.warning("Batch contains invalid values - skipping")
            return None, None
            
        return images_tensor, list(prompts)
    except Exception as e:
        logger.warning(f"Error collating batch: {str(e)}")
        return None, None

def inject_lora(unet, r=4):
    """LoRA injection with validation"""
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

def train_lora(zip_path, output_dir):
    """Robust training function with enhanced stability"""
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision='fp16'
    )
    set_seed(42)
    
    # Enable memory efficient attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Load pipeline with error handling
    try:
        logger.info("Loading Stable Diffusion pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(accelerator.device)
        
        # Use DDPM scheduler
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        raise

    # Freeze components
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Load text components
    try:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
        text_encoder = text_encoder.to(dtype=torch.float16)
    except Exception as e:
        logger.error(f"Failed to load text components: {str(e)}")
        raise

    # Inject LoRA
    inject_lora(pipe.unet)

    # Prepare dataset with validation
    try:
        logger.info("Preparing dataset...")
        full_dataset = RobustTwoActionDataset(zip_path)
        
        # Analyze dataset
        logger.info(f"Total valid samples: {len(full_dataset)}")
        if len(full_dataset) == 0:
            raise ValueError("No valid samples found in dataset")
            
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            collate_fn=safe_collate,
            persistent_workers=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=safe_collate,
            persistent_workers=True
        )
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=5e-5,  # Reduced learning rate
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=len(train_dataloader) * 5,  # 5 epochs
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Prepare with accelerator
    pipe.unet, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    # Training loop with enhanced stability
    num_epochs = 5
    best_loss = float('inf')
    loss_history = []
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        pipe.unet.train()
        epoch_loss = 0
        valid_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None or batch[0] is None:
                logger.debug(f"Skipping invalid batch at step {step}")
                continue

            images, captions = batch
            images = images.to(accelerator.device, dtype=torch.float16)

            with torch.no_grad():
                # Encode images to latents with validation
                latents = pipe.vae.encode(images).latent_dist.sample()
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    logger.warning(f"Invalid latents at step {step}, skipping")
                    continue
                latents = latents * 0.18215

                # Encode text
                input_ids = tokenizer(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state
                if torch.isnan(encoder_hidden_states).any() or torch.isinf(encoder_hidden_states).any():
                    logger.warning(f"Invalid text embeddings at step {step}, skipping")
                    continue

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, 
                (latents.shape[0],), 
                device=latents.device
            ).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise with mixed precision
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

            # Skip invalid losses
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"Invalid loss at step {step}, skipping")
                optimizer.zero_grad()
                continue

            # Backprop with gradient scaling and clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            
            # Log progress
            if step % 10 == 0:
                avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step}/{len(train_dataloader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                )

        # Calculate epoch metrics
        if valid_batches > 0:
            avg_epoch_loss = epoch_loss / valid_batches
            loss_history.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} Avg Train Loss: {avg_epoch_loss:.4f}")
        else:
            logger.warning(f"No valid batches in epoch {epoch+1}")
            continue

        # Validation
        pipe.unet.eval()
        test_loss = 0
        test_batches = 0
        with torch.no_grad():
            for batch in test_dataloader:
                if batch is None or batch[0] is None:
                    continue
                    
                images, captions = batch
                images = images.to(accelerator.device, dtype=torch.float16)
                
                # Encode images and text
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
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
                
                # Predict and calculate loss
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
                test_loss += loss.item()
                test_batches += 1

        if test_batches > 0:
            avg_test_loss = test_loss / test_batches
            logger.info(f"Epoch {epoch+1} Avg Test Loss: {avg_test_loss:.4f}")
            
            # Save best model
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                if accelerator.is_main_process:
                    save_path = os.path.join(output_dir, "best_model")
                    pipe.save_pretrained(save_path, safe_serialization=True)
                    logger.info(f"Saved best model with test loss: {best_loss:.4f}")
        else:
            logger.warning("No valid test batches")

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = os.path.join(output_dir, "final_model")
        pipe.save_pretrained(final_save_path, safe_serialization=True)
        logger.info(f"Training complete! Final model saved at: {final_save_path}")
        
        # Save loss history
        np.save(os.path.join(output_dir, "loss_history.npy"), np.array(loss_history))

if __name__ == "__main__":
    # Paths - update these for your server
    zip_path = "/home/ubuntu/money-leads-video-generator/Dataset2.zip"
    output_dir = "/home/ubuntu/money-leads-video-generator/lora_model"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify dataset exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip file not found at {zip_path}")
    
    # Start training
    try:
        train_lora(zip_path, output_dir)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
