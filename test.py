import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "D:/viot/notebook/best_leaf_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Healthy", "Dry Leaf", "Unhealthy"]

# Specific disease mapping for Unhealthy class
UNHEALTHY_DISEASES = {
    "Apple": ["Apple_scab", "Anthracnose", "Black_rot", "Cedar_apple_rust"],
    "Tomato": ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
               "Septoria_leaf_spot", "Spider_mites", "Target_Spot", "Tomato_mosaic_virus",
               "Tomato_Yellow_Leaf_Curl_Virus"],
    "Potato": ["Early_blight", "Late_blight"],
    "Litchi": ["Leaf_blight", "Algal_spot", "Anthracnose", "Leaf_mites",
               "Mayetiola", "Leaf_Gall", "Leaf_Holes", "Blight", "Fungal_Leaf_Spot"],
    "Pepper": ["Bacterial_spot"],
    "Citrus": ["Canker"],
    "Cherry": ["Powdery_mildew"]
}

# ==============================
# MODEL
# ==============================
model = models.mobilenet_v3_small(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        pred_class = CLASS_NAMES[pred.item()]

    if pred_class == "Unhealthy":
        # Here you can detect the plant type if you have multiple plant models
        # For simplicity, randomly choose one plant
        import random
        plant = random.choice(list(UNHEALTHY_DISEASES.keys()))
        disease_list = UNHEALTHY_DISEASES[plant]
        return f"{pred_class}: {', '.join(disease_list)}"
    return pred_class

# ==============================
# GUI
# ==============================
def select_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not path:
        return
    try:
        img = Image.open(path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        result = predict_image(path)
        result_label.config(text=result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Leaf Health Predictor")

btn = tk.Button(root, text="Select Leaf Image", command=select_image)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
