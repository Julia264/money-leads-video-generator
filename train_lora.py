import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from lora_diffusion import inject_trainable_lora
from diffusers.training_utils import set_seed

# 🟡 Dataset
class FrameDataset(Dataset):
    def __init__(self, root_dir, prompt_dict, image_size=512):
        self.samples = []
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                frames = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith(".png")]
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
        image = self.transform(Image.open(path).convert("RGB"))
        return image, prompt

# 🟢 LoRA Injection
def inject_lora(unet, r=4):
    inject_trainable_lora(
        unet,
        r=r,
        target_replace_module=["CrossAttention", "Attention"],
    )

# 🔵 Training
def train_lora(data_dir, prompts, output_dir):
    accelerator = Accelerator()
    set_seed(42)

    # Load pipeline with float16 precision
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(accelerator.device)

    # Load text encoder and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
    text_encoder = text_encoder.to(dtype=torch.float16)  # Match UNet dtype

    # Inject LoRA layers
    inject_lora(pipe.unet)

    # Prepare dataset and dataloader
    dataset = FrameDataset(data_dir, prompts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Optimizer
    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)

    # Prepare with accelerator
    pipe.unet, optimizer, dataloader = accelerator.prepare(
        pipe.unet, optimizer, dataloader
    )

    pipe.unet.train()

    for epoch in range(3):
        for step, (images, captions) in enumerate(dataloader):
            with accelerator.accumulate(pipe.unet):
                # Move images to device and ensure float16
                images = images.to(accelerator.device, dtype=torch.float16)

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
                
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device)).last_hidden_state

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                model_pred = pipe.unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")

    # Save model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipe.save_pretrained(output_dir)
        print("✅ Training complete! Model saved at:", output_dir)

# 🔥 Main
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

    train_lora(data_dir, prompts, output_dir)
