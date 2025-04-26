# train_lora.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AnimateDiffPipeline
from peft import get_peft_model, LoraConfig
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator

class MotionFrameDataset(Dataset):
    def __init__(self, root_dir, prompt_dict, image_size=512):
        self.samples = []
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                frames = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith(".png")]
                for f in frames:
                    self.samples.append((f, prompt_dict.get(label, "a person moving")))

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

def train_lora(data_dir, prompts, output_dir):
    accelerator = Accelerator()

    # Load the base AnimateDiff model
pipe = AnimateDiffPipeline.from_pretrained(
    pretrained_model_path="./models/mm_sd_v14.ckpt",
    motion_module_path="./models/mm_sd_v15.ckpt",
    torch_dtype=torch.float16
).to(accelerator.device)


    # Apply LoRA to the UNet
config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="UNET"
    )
    pipe.unet = get_peft_model(pipe.unet, config)
    pipe.unet.train()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)

    dataset = MotionFrameDataset(data_dir, prompts)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-5)

    for epoch in range(5):
        for i, (images, texts) in enumerate(dataloader):
            with accelerator.accumulate(pipe.unet):
                input_ids = tokenizer(list(texts), padding="max_length", truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(images)
                noisy = images + 0.1 * noise
                outputs = pipe.unet(noisy, encoder_hidden_states=encoder_hidden_states)

                loss = torch.nn.functional.mse_loss(outputs.sample, images)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}] Step {i}: Loss = {loss.item():.4f}")

    accelerator.wait_for_everyone()
    pipe.save_pretrained(output_dir)
    print("✅ LoRA Fine-Tuning Complete")

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    prompts = {
        "احبك": "a person saying 'I love you' warmly",
        "احسنت": "a person saying 'Well done' and smiling",
        "اعجبني": "a person showing approval with a nod",
        "انت عظيم": "a person cheering 'You're amazing'",
        "تصفيق": "a person clapping joyfully",
        "حبيبي": "a person saying 'my dear' affectionately",
        "مرحبا": "a person waving hello",
        "هذا رائع": "a person excited saying 'That’s great!'",
        "واو": "a person amazed saying 'Wow!'",
        "مدهش": "a person expressing 'amazing' happily"
    }
    data_dir = os.path.join(BASE_DIR, "datasets", "الحركات")
    output_dir = os.path.join(BASE_DIR, "models", "fine-tuned-motion")

    train_lora(data_dir, prompts, output_dir)
