from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

# 1. Charger le modèle
# Vous pouvez spécifier n'importe quel modèle Stable Diffusion compatible Img2Img
# Assurez-vous d'avoir suffisamment de RAM GPU pour charger le modèle
model_id_or_path = "runwayml/stable-diffusion-v1-5" # Ou un autre modèle que vous avez téléchargé
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

# 2. Charger votre image d'entrée
# Remplacez 'votre_image_entree.jpg' par le chemin de votre image
try:
    init_image = Image.open("Entree.jpg").convert("RGB")
    print(init_image)


# Vous pouvez redimensionner l'image si nécessaire,
# mais Stable Diffusion fonctionne généralement bien avec des résolutions 512x512 ou 768x512, 1024x1024 etc.
# Si votre image est trop grande, cela consommera beaucoup de VRAM.
init_image = init_image.resize((768, 512)) # Exemple de redimensionnement

# 3. Définir le prompt
prompt = "Un château médiéval fantastique, sous un ciel orageux, style peinture à l'huile dramatique"

# 4. Générer l'image modifiée
# denoising_strength : 0.0 (pas de changement) à 1.0 (régénération complète)
# Une valeur entre 0.6 et 0.8 est un bon point de départ pour des transformations significatives.
generated_image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75, # Force de débruitage
    guidance_scale=7.5 # Influence du prompt sur la génération
).images[0]

# 5. Sauvegarder ou afficher l'image résultante
generated_image.save("image_generee_img2img.png")
print("Image générée et sauvegardée sous 'image_generee_img2img.png'")