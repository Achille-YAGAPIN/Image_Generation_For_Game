import os
import subprocess

# --- CONFIGURATION DES CHEMINS ---
# Le dossier de base est le dossier 'scripts'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# La racine du projet est un niveau au-dessus de 'scripts'
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Dossiers et Fichiers
DATASET_DIR = os.path.join(ROOT_DIR, 'data', 'images')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'models')
CAPTION_FILE = os.path.join(ROOT_DIR, 'metadata.jsonl')

# CHEMIN CORRIGÉ VERS LE SCRIPT D'ENTRAÎNEMENT 
# (Suppose que vous avez téléchargé 'train_text_to_image_lora.py' dans /scripts/)
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'train_text_to_image_lora.py') 

# --- HYPERPARAMÈTRES ET MODÈLE ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5" # Le modèle Stable Diffusion de base

RESOLUTION = 512 # Taille des images traitées (Doit correspondre à preprocess.py)
TRAIN_BATCH_SIZE = 1 
GRADIENT_ACCUMULATION_STEPS = 4 # Simule une batch size de 4 (1 * 4)
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 5000 
SAVE_STEPS = 500 # Sauvegarder un checkpoint tous les 500 étapes
RANK = 4 # Le rang LoRA (paramètre de compression)

def prepare_and_check_paths():
    """Vérifie l'existence des données et crée le dossier de sortie."""
    
    # 1. Vérifier la présence du script d'entraînement
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Erreur : Le script d'entraînement {TRAIN_SCRIPT} est introuvable.")
        print("Avertissement : Veuillez télécharger train_text_to_image_lora.py et le placer dans le dossier /scripts.")
        return False
        
    # 2. Vérifier la présence des données
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"Erreur : Le dossier de données {DATASET_DIR} est vide ou n'existe pas.")
        print("Veuillez d'abord exécuter preprocess.py.")
        return False
    
    # 3. Vérifier la présence des annotations
    if not os.path.exists(CAPTION_FILE):
        print(f"Erreur : Le fichier d'annotations {CAPTION_FILE} est introuvable.")
        return False
        
    # 4. Créer le dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return True

def run_training():
    """Exécute l'entraînement LoRA en utilisant accelerate launch."""
    
    if not prepare_and_check_paths():
        return
        
    print("\n" + "="*50)
    print(f"Lancement de l'entraînement LoRA sur {MODEL_NAME}...")
    print(f"Script utilisé : {TRAIN_SCRIPT}")
    print("="*50 + "\n")

    # Construction de la commande 'accelerate launch'
    command = [
    "accelerate", "launch", 
    TRAIN_SCRIPT,
    "--pretrained_model_name_or_path", MODEL_NAME,
    "--train_data_dir", DATASET_DIR, 
    "--caption_column","text",
    "--output_dir", OUTPUT_DIR,
    "--resolution", str(RESOLUTION),
    # "--center_crop", # L'absence de l'argument le laisse à False par défaut (RandomCrop)
    "--train_batch_size", str(TRAIN_BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
    "--learning_rate", str(LEARNING_RATE),
    "--lr_scheduler", "constant", 
    "--lr_warmup_steps", "0",
    "--max_train_steps", str(MAX_TRAIN_STEPS),
    "--checkpointing_steps", str(SAVE_STEPS), # Utilisation de l'argument du script text-to-image
    "--validation_epochs", "1", # Ajout pour s'assurer qu'il y a une validation
    "--seed", "42",
    "--mixed_precision", "fp16", 
    "--rank", str(RANK), # Argument corrigé
    "--dataloader_num_workers", "8",
    # --tokenizer_name et --caption_file SONT SUPPRIMÉS car non supportés
    # "--report_to", "wandb",
]

    try:
        # Exécution de la commande
        subprocess.run(command, check=True)
        print("\n" + "="*50)
        print("✅ Entraînement LoRA terminé avec succès!")
        print(f"Les LoRA sont dans le dossier : {OUTPUT_DIR}")
        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors de l'exécution de l'entraînement : {e}")
        print("Vérifiez les messages d'erreur détaillés ci-dessus.")
    except FileNotFoundError:
        print("\n❌ Erreur : 'accelerate' ou 'python' non trouvé. Vérifiez l'activation de l'environnement Conda.")


if __name__ == "__main__":
    run_training()
