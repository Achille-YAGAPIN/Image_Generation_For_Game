import os
import json
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
DATA_IMAGES_DIR = os.path.join(ROOT_DIR, 'data', 'images')
CAPTION_FILE = os.path.join(ROOT_DIR, 'caption.json')

TARGET_SIZE = (512, 512)
BACKGROUND_COLOR = (255, 255, 255) 

def process_image(input_path: str, output_path: str):
    try:
        img = Image.open(input_path).convert("RGBA")
        
        img_background = Image.new('RGB', img.size, BACKGROUND_COLOR)
        img_background.paste(img, mask=img.getchannel("A"))

        resize_img = img_background.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        resize_img.save(output_path, 'JPEG',quality=95)
        return True
    except Exception as e:
        print(f"Error during image processing {input_path} : {e}")
        return False

def load_annotation(__file__ caption_file):
    try:
        with open(caption_file, 'r') as f:
            annotations = json.load(f)
        print(f"Annotations chargées depuis {caption_file}.")
    except (FileNotFoundError, json.JSONDecodeError) as e :
        print(f"Erreur lors du chargement de {caption_file} : {e}")
        return

def main():
    # 1. Charger les annotations
    
    load_annotation(CAPTION_FILE)

    success_count = 0

    # 2. Traiter chaque entrée
    for item in annotations:
        relative_path = item.get("image") # Ex: "foot/evobattle-player-0-left-foot-0.png"

        if not relative_path:
            print("Avertissement : Entrée JSON sans clé 'image'. Ignorée.")
            continue

        # Chemin complet vers l'image source
        input_file = os.path.join(ASSETS_DIR, relative_path)

        # Construire le chemin de sortie (on garde le nom mais on change l'extension en .jpg)
        # On remplace l'extension par .jpg
        output_filename = os.path.splitext(os.path.basename(relative_path))[0] + ".jpg"
        output_file = os.path.join(DATA_IMAGES_DIR, output_filename)

        # Traitement
        if os.path.exists(input_file):
            if process_image(input_file, output_file):
                success_count += 1
        else:
            print(f"Avertissement : Fichier source non trouvé : {input_file}")

    print("-" * 30)
    print(f"Traitement terminé. {success_count} images traitées avec succès et sauvegardées dans /data/images/")

if __name__ == "__main__":
    main()