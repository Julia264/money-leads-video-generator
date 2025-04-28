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
        """
        Dataset class for the two actions (clapping, waving) from Dataset2.zip
        
        Args:
            zip_path: Path to Dataset2.zip
            image_size: Ignored here because .pt tensors are loaded directly
        """
        self.samples = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            clapping_files = [f for f in file_list if 'clapping' in f.lower() and f.lower().endswith('.pt')]
            for pt_file in clapping_files:
                self.samples.append((pt_file, "a person clapping hands"))
            
            waving_files = [f for f in file_list if 'waving' in f.lower() and f.lower().endswith('.pt')]
            for pt_file in waving_files:
                self.samples.append((pt_file, "a person waving hello"))

        self.zip_path = zip_path  # Save zip path for loading in __getitem__

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, prompt = self.samples[idx]
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(file_name) as pt_data:
                    # Load tensor
                    tensor = torch.load(pt_data, map_location="cpu")
                    
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # Expand grayscale to 3 channels
                    elif tensor.shape[0] != 3:
                        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
                    
                    # Normalize manually [-1, 1] because no PIL transform
                    tensor = (tensor - 0.5) / 0.5

            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning("Invalid tensor values detected")
                return None, None

            return tensor, prompt
        except Exception as e:
            logger.warning(f"Error loading tensor: {str(e)}")
            return None, None


def safe_collate(batch):
    """Collate function that filters out None samples"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    images, prompts = zip(*batch)
    return torch.stack(images), list(prompts)

def inject_lora(unet, r=4):
    """Inject LoRA layers into UNet"""
    inject_trainable_lora(
        unet,
        r=r,
        target_replace_module=["CrossAttention", "Attention"],
    )

def train_lora(zip_path, output_dir):
    """Train LoRA model on the two-action dataset"""
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=os.path.join(output_dir, "logs")
    )
    set_seed(42)
    
    # Enable memory efficient attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Load pipeline
    logger.info("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(accelerator.device)
    
    # Use DDPM scheduler
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze all components except LoRA
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Load text components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
    #text_encoder = text_encoder.to(dtype=torch.float16)

    # Inject LoRA
    logger.info("Injecting LoRA layers...")
    inject_lora(pipe.unet)

    # Prepare dataset and dataloader
    logger.info("Preparing dataset...")
    dataset = TwoActionDataset(zip_path)
    
    # Create train/test split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=safe_collate
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=safe_collate
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_dataloader)*5,  # 5 epochs
        eta_min=1e-5
    )

    # Prepare with accelerator
    pipe.unet, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    # Training loop
    num_epochs = 5
    best_loss = float('inf')
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        pipe.unet.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None or batch[0] is None:
                continue

            images, captions = batch
            images = images.to(accelerator.device)

            with torch.no_grad():
                # Encode images to latents
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                
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

            # Predict noise
            with accelerator.autocast():
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

            # Skip NaN losses
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"Invalid loss at step {step}, skipping")
                optimizer.zero_grad()
                continue

            # Backprop
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(pipe.unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
            # Log progress
            if step % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step}/{len(train_dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        # Validation
        pipe.unet.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                if batch is None or batch[0] is None:
                    continue
                    
                images, captions = batch
                images = images.to(accelerator.device)
                
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

        avg_test_loss = test_loss / len(test_dataloader)
        logger.info(f"Epoch {epoch+1} Test Loss: {avg_test_loss:.4f}")
        
        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            if accelerator.is_main_process:
                save_path = os.path.join(output_dir, "best_model")
                pipe.save_pretrained(save_path)
                logger.info(f"Saved best model with test loss: {best_loss:.4f}")

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = os.path.join(output_dir, "final_model")
        pipe.save_pretrained(final_save_path)
        logger.info(f"Training complete! Model saved at: {final_save_path}")

if __name__ == "__main__":
    # Paths
    zip_path = "/home/ubuntu/money-leads-video-generator/Dataset2.zip"  # Update this path
    output_dir = "/home/ubuntu/money-leads-video-generator/models"  # Update this path
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify zip file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip file not found at {zip_path}")
    
    # Start training
    train_lora(zip_path, output_dir)
