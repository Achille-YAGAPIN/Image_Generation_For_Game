import os
import json
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
DATA_IMAGES_DIR = os.path.join(ROOT_DIR, 'data', 'images')
CAPTION_FILE = os.path.join(ROOT_DIR, 'metadata.jsonl')

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

import json

def load_annotation(caption_file: str):
    try:
        annotations = []
        with open(caption_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        annotations.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {line[:50]}... : {e}")
        
        print(f"Annotations loaded from {caption_file}. Total: {len(annotations)} items")
        return annotations
    
    except FileNotFoundError as e:
        print(f"Error during loading of {caption_file} : {e}")
        return []

def load_item(item: str):
    relative_path = item.get("image") 
    if not relative_path:
        print("Warning: JSON input without key 'image'.Ignored")
        return False

    input_file = os.path.join(ASSETS_DIR, relative_path)
    
    output_file = os.path.join(DATA_IMAGES_DIR, relative_path)
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.splitext(output_file)[0] + ".jpg"
    return process_image(input_file, output_file)


def main():
    annotations = load_annotation(CAPTION_FILE)
    if not annotations:
        print("Aucune annotation trouvée. Fin du script.")
        return

    success_count = 0
    for item in annotations:
        if load_item(item):
            success_count += 1

    print("-" * 30)
    print(f"Processing complete. {success_count} images successfully processed and saved in /data/images/")

if __name__ == "__main__":
    main()