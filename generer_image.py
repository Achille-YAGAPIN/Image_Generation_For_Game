from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

HF_TOKEN = "" #TODO mettre le  token plus tard dans le git

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    use_safetensors=True
).to("cuda")

pipe.enable_attention_slicing()

# ---- Chargement du LoRA ----
# Option 1 : LoRA venant de HuggingFace Hub
pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-FLUX.1-dev-8steps-lora.safetensors")

# Option 2 : LoRA local
# pipe.load_lora_weights("./mon_lora", weight_name="lora.safetensors")

# Ajuster l’influence du LoRA
pipe.fuse_lora(lora_scale=0.75)

prompt = "A dragon-dog hybrid spitting computers"

image = pipe(prompt, guidance_scale=7.5).images[0]

image.save("ma_deuxieme_image.png")

