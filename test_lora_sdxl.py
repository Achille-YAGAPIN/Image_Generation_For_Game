import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from PIL import Image

# Charger le modèle SDXL
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype = torch.float16,
    use_safetensors=True
).to("cuda")

# Charger un LoRA depuis HuggingFace Hub
pipe.load_lora_weights(
    ".",
    weight_name="minecraft.safetensors"
)

# Fusionner le LoRA avec une intensité donnée
pipe.fuse_lora(lora_scale=0.75)

prompt = "A dragon with a castle"

# LCM = low guidance
image = pipe(prompt, guidance_scale=9).images[0]

image.save("ma_premiere_image.png")
print("Image saved as ma_premiere_image.png")

