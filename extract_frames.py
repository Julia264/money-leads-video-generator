!pip install diffusers transformers accelerate peft bitsandbytes --upgrade

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator

# Custom Dataset
class FrameDataset(Dataset):
    def __init__(self, root_dir, prompt_dict, image_size=512):
        self.samples = []
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                frames = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith(".png")]
                for f in frames:
                    self.samples.append((f, prompt_dict.get(label, "a photo of a person")))

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

# Training function
def train_lora(data_dir, prompts, output_dir):
    accelerator = Accelerator()
    
    model_id = "CompVis/stable-diffusion-v1-5"

    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        revision="fp16",
    ).to(accelerator.device)

    # Apply LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["attn1", "attn2", "proj_in", "proj_out"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"  # Suitable for diffusion-based UNet blocks
    )
    pipe.unet = get_peft_model(pipe.unet, config)
    pipe.unet.train()

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(accelerator.device)

    dataset = FrameDataset(data_dir, prompts)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)

    for epoch in range(3):
        for step, (images, captions) in enumerate(dataloader):
            with accelerator.accumulate(pipe.unet):
                input_ids = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

                noise = torch.randn_like(images)
                timesteps = torch.randint(0, 1000, (images.shape[0],), device=images.device).long()

                noisy_images = images + 0.1 * noise

                model_pred = pipe.unet(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                loss = torch.nn.functional.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch+1} Step {step}: Loss = {loss.item():.4f}")

    accelerator.wait_for_everyone()
    pipe.save_pretrained(output_dir)
    print("✅ LoRA training complete and model saved!")

# Main block
if __name__ == "__main__":
    BASE_DIR = os.getcwd()

    prompts = {
        "احبك": "a person saying I love you",
        "احسنت": "a person saying Well done happily",
        "اعجبني": "a person showing like gesture",
        "انت عظيم": "a person excited shouting You're amazing",
        "تصفيق": "a person clapping hands",
        "حبيبي": "a person saying my dear with love",
        "مرحبا": "a person waving hand",
        "هذا رائع": "a person saying That's great with excitement",
        "واو": "a person amazed saying Wow",
        "مدهش": "a person surprised saying Amazing"
    }

    data_dir = os.path.join(BASE_DIR, "datasets", "frames")
    output_dir = os.path.join(BASE_DIR, "lora_finetuned_model")

    train_lora(data_dir, prompts, output_dir)
