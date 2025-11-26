from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

HF_TOKEN = "" #TODO mettre le  token plus tard dans le git

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

pipe.enable_attention_slicing()

# ---- Chargement du LoRA ----
# Option 1 : LoRA venant de HuggingFace Hub
pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-SDXL-2steps-lora.safetensors")
pipe.load_lora_weights(".", weight_name="minecraft.safetensors")

# Ajuster l’influence du LoRA
pipe.unet_lora_scales = {
    "Hyper-SD": 0.8,      # 80% d’effet pour l’UNet
    "minecraft": 0.5      # 50% d’effet pour l’UNet
}


prompt = "A dragon-dog hybrid spitting computers, pixel art"

image = pipe(prompt, guidance_scale=7.5).images[0]

image.save("ma_deuxieme_image.png")

