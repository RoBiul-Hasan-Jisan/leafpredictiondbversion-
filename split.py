import os
import shutil
import random

# ==============================
# 1. CONFIG
# ==============================
DATA_ROOT = r"D:/viot/Data"          # Original dataset folder
OUTPUT_ROOT = r"D:/viot/Data__Split" # Output split folder
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

# ==============================
# 2. COARSE & PLANT MAPPINGS
# ==============================
def clean_name(name):
    """Normalize folder/class names"""
    return name.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

COARSE_MAPPING = {
    "Healthy": [clean_name(x) for x in [
        "Healthy", "Healthy Leaf", "Apple___healthy", "Blueberry___healthy", 
        "Cherry_including_sour___healthy", "Tomato_healthy", "Potato___healthy", "Pepper__bell___healthy"]],
    "Dry": [clean_name(x) for x in ["Dry Leaf"]],
    "Unhealthy": []
}

PLANT_MAPPING = {
    "Apple": [clean_name(x) for x in ["Apple___Apple_scab", "Anthracnose", "Apple___Black_rot", "Apple___Cedar_apple_rust"]],
    "Tomato": [clean_name(x) for x in [
        "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
        "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
        "Tomato_Target_Spot", "Tomato_Tomato_mosaic_virus", "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Septoria_spot",
        "Spider_mites", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato_Spider_mites Two-spotted_spider_mite"
    ]],
    "Potato": [clean_name(x) for x in ["Potato___Early_blight", "Potato___Late_blight"]],
    "Litchi": [clean_name(x) for x in [
        "Leaf_blight_Litchi_leaf_diseases","Litchi_algal_spot_in_non-direct_sunlight",
        "Litchi_anthracnose_on_cloudy_day","Litchi_leaf_mites_in_direct_sunlight",
        "Litchi_mayetiola_after_raining","Leaf Gall","Leaf Holes","Leaf Blight",
        "Fungal Leaf Spot","Felt Leaf","Deficiency Leaf","Curl Leaf","Bituminous Leaf","Anthrax Leaf"
    ]],
    "Pepper": [clean_name(x) for x in ["Pepper__bell___Bacterial_spot"]],
    "Citrus": [clean_name(x) for x in ["Citrus_Canker"]],
    "Cherry": [clean_name(x) for x in ["Cherry_including_sour___Powdery_mildew"]],
    "Others": []
}

# ==============================
# 3. HELPERS
# ==============================
def get_coarse_class(class_name):
    for coarse, sublist in COARSE_MAPPING.items():
        if class_name in sublist:
            return coarse
    return "Unhealthy"

def get_plant_class(class_name):
    for plant, sublist in PLANT_MAPPING.items():
        if class_name in sublist:
            return plant
    return "Others"

random.seed(SEED)

# ==============================
# 4. CREATE BASE SPLIT FOLDERS
# ==============================
for split in SPLIT_RATIOS.keys():
    os.makedirs(os.path.join(OUTPUT_ROOT, split), exist_ok=True)

# ==============================
# 5. PROCESS DATA
# ==============================
folders = os.listdir(DATA_ROOT)
folders.sort()

for folder in folders:
    folder_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    class_name = clean_name(folder)
    images = [f for f in os.listdir(folder_path)
              if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]

    if len(images) == 0:
        print(f"[WARNING] No images found in {folder_path}")
        continue

    print(f"[INFO] Processing {class_name}: {len(images)} images", flush=True)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    split_mapping = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, split_images in split_mapping.items():
        coarse_class = get_coarse_class(class_name)
        if coarse_class == "Unhealthy":
            plant_class = get_plant_class(class_name)
            split_class_path = os.path.join(OUTPUT_ROOT, split, coarse_class, plant_class, class_name)
        else:
            split_class_path = os.path.join(OUTPUT_ROOT, split, coarse_class, class_name)

        os.makedirs(split_class_path, exist_ok=True)

        for img in split_images:
            src = os.path.join(folder_path, img)
            dst = os.path.join(split_class_path, img)
            shutil.copy2(src, dst)

print("[DONE] Dataset split into train/val/test with coarse and plant-specific hierarchy!")
