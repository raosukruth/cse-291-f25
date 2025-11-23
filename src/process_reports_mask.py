import os
import numpy as np
from PIL import Image
import audio

BASE_DIR = "/home/sjrao/cse291"
OUTPUTS_DIR = BASE_DIR + "/outputs"
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

GENRE_OUTPUT_DIRS = {}
for i in GENRES:
    GENRE_OUTPUT_DIRS[i] = {"center": os.path.join(OUTPUTS_DIR, f"{i}_center_masks"), 
                            "gaussian": os.path.join(OUTPUTS_DIR, f"{i}_gaussian_masks"),
                            "low_dim": os.path.join(OUTPUTS_DIR, f"{i}_low_dim_masks")}

    for k in GENRE_OUTPUT_DIRS[i].values():
        os.makedirs(k, exist_ok=True)

def get_images_list(dir_name):
    files = os.listdir(dir_name)
    files.sort()
    return files

def generate_masks(image_path):
    _, center_mask, _gaussian_mask, low_dim = audio.get_distorted_all(image_path, None, None,
                                                                         None, None, visualize_result=False)
    return {"center": center_mask, "gaussian": _gaussian_mask, "low_dim": low_dim}

def save_masks(image_name, genre, masks):
    base = os.path.splitext(image_name)[0]
    paths = {"genre": genre, "base_name": base}
    for k, img in masks.items():
        output_path = GENRE_OUTPUT_DIRS[genre][k] + "/" + base + ".png"
        if isinstance(img, Image.Image):
            img.save(output_path)
        else:
            Image.fromarray(np.asarray(img).astype(np.uint8)).save(output_path)
        paths[f"{k}_path"] = output_path
    return paths

if __name__ == "__main__":
    for genre in GENRES:
        image_dir = BASE_DIR + "/GTZAN/images_original" + "/" + genre
        images = get_images_list(image_dir)
        print(f"Processing {len(images)} images for genre '{genre}'")
        for idx, img_name in enumerate(images):
            image_path = image_dir + "/" + img_name
            masks = generate_masks(image_path)
            save_masks(img_name, genre, masks)

    print("FINISHED MASKING ALL IMAGES")