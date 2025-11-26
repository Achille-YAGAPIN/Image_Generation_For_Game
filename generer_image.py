from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

HF_TOKEN = "" #TODO mettre le  token plus tard dans le git

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()

# ---- Chargement du LoRA ----
# Option 1 : LoRA venant de HuggingFace Hub
pipe.load_attn_procs("ostris/SD-1.5-Lora-Examples", weight_name="armor.safetensors")

# Option 2 : LoRA local
# pipe.load_lora_weights("./mon_lora", weight_name="lora.safetensors")

# Ajuster l’influence du LoRA
pipe.fuse_lora(lora_scale=0.75)

prompt = "A dragon-dog hybrid spitting computer code, pixel art style"

image = pipe(prompt, guidance_scale=7.5).images[0]

image.save("ma_premiere_image.png")

