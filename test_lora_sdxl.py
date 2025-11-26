import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from PIL import Image

# Charger le modèle SDXL
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Remplacer le scheduler (obligatoire pour un LoRA LCM)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Charger un LoRA depuis HuggingFace Hub
pipe.load_lora_weights(
    "latent-consistency/lcm-lora-sdxl",
    weight_name="pytorch_lora_weights.safetensors"
)

# Fusionner le LoRA avec une intensité donnée
pipe.fuse_lora(lora_scale=0.75)

prompt = "A dragon-dog hybrid spitting computers"

# LCM = low guidance
image = pipe(prompt, guidance_scale=1.0, num_inference_steps=6).images[0]

image.save("ma_premiere_image.png")
print("Image saved as ma_premiere_image.png")

