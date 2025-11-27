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
BACKGROUND_COLOR = (255, 255, 255) # Blanc (standard pour masquer la transparence)

def setup_directories():
    """Crée le dossier de sortie s'il n'existe pas."""
    os.makedirs(DATA_IMAGES_DIR, exist_ok=True)
    print(f"Dossier de sortie créé/vérifié : {DATA_IMAGES_DIR}")

def process_image(input_path: str, output_path: str):
    """
    Charge une image, la gère la transparence, la redimensionne et la sauvegarde.
    """
    try:
        # 1. Ouvrir l'image
        img = Image.open(input_path).convert("RGBA")
        
        # 2. Gérer la transparence (Remplacer le canal alpha par un fond blanc)
        # Créer une image blanche de même taille
        background = Image.new('RGB', img.size, BACKGROUND_COLOR)
        
        # Superposer l'image originale sur le fond blanc en utilisant le canal alpha comme masque
        # On utilise le canal A (alpha) de l'image originale comme masque.
        background.paste(img, mask=img.split()[3])
        
        # Convertir l'image résultante en mode RGB
        processed_img = background.convert('RGB')
        
        # 3. Redimensionner (en utilisant une méthode de haute qualité)
        # Utiliser THUMBNAIL pour redimensionner sans déformer/recadrer
        processed_img = processed_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # 4. Sauvegarder l'image traitée dans le dossier /data/images
        processed_img.save(output_path, 'JPEG',quality=95)
        
        return True

    except Exception as e:
        print(f"Erreur lors du traitement de l'image {input_path} : {e}")
        return False

def main():
    """Fonction principale pour parcourir les annotations et traiter les images."""
    setup_directories()
    
    # 1. Charger les annotations
    try:
        with open(CAPTION_FILE, 'r') as f:
            annotations = json.load(f)
        print(f"Annotations chargées depuis {CAPTION_FILE}. Total : {len(annotations)}")
    except FileNotFoundError:
        print(f"Erreur : Le fichier d'annotations {CAPTION_FILE} est introuvable.")
        return
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier {CAPTION_FILE} n'est pas un JSON valide.")
        return
    
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
