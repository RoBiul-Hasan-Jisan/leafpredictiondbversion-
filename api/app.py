from flask import Flask, request, jsonify
from flask_cors import CORS  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes

# ==============================
# CONFIG
# ==============================
LEAF_MODEL_PATH = r"D:\viot\notebook\leaf_detector.pth"
DISEASE_MODEL_PATH = r"D:\viot\notebook\best_cpu_model.pth"
STAGE2_MODEL_PATH = r"D:\viot\notebook\stage2_best_model.pth"

DEVICE = torch.device("cpu")
LEAF_CLASS_NAMES = ["Leaf", "Not Leaf"]
DISEASE_CLASS_NAMES = ["Dry", "Healthy", "Unhealthy"]
STAGE2_CLASS_NAMES = [
    "Apple", "Cherry", "Citrus", "Litchi", "Others", "Pepper", "Potato", "Tomato"
]

THRESHOLD_LEAF = 0.8
THRESHOLD_UNHEALTHY = 0.6

CATEGORY_DISEASES = {
    "Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust"],
    "Cherry": ["Brown Rot"],
    "Citrus": ["Citrus Canker"],
    "Litchi": [
        "Anthrax Leaf", "Bituminous Leaf", "Curl Leaf", "Deficiency Leaf",
        "Felt Leaf", "Fungal Leaf Spot", "Leaf Blight", "Leaf blight litchi",
        "Leaf Gall", "Leaf Holes", "Algal Spot", "Anthracnose", "Leaf Mites",
        "Litchi mayetiola after raining"
    ],
    "Pepper": ["Bacterial Spot"],
    "Potato": ["Early Blight", "Late Blight"],
    "Tomato": [
        "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
        "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Target Spot",
        "Tomato Tomato Mosaic Virus", "Tomato YellowLeaf Curl Virus"
    ],
    "Others": []
}

# ==============================
# LOAD MODELS
# ==============================
def load_leaf_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LEAF_CLASS_NAMES))
    model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_disease_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(DISEASE_CLASS_NAMES))
    )
    model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_stage2_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(STAGE2_CLASS_NAMES))
    )
    model.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

leaf_model = load_leaf_model()
disease_model = load_disease_model()
stage2_model = load_stage2_model()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==============================
# FLASK ENDPOINT
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    response = {}

    # STEP 1: Leaf detection
    with torch.no_grad():
        out_leaf = leaf_model(img_tensor)
        probs_leaf = F.softmax(out_leaf, dim=1)[0]
        max_prob_leaf, pred_idx_leaf = torch.max(probs_leaf, 0)

    response["is_leaf"] = LEAF_CLASS_NAMES[pred_idx_leaf.item()] == "Leaf"
    response["leaf_confidence"] = float(max_prob_leaf.item())

    if not response["is_leaf"] and max_prob_leaf.item() > THRESHOLD_LEAF:
        return jsonify(response)

    # STEP 2: Health prediction
    with torch.no_grad():
        out_health = disease_model(img_tensor)
        probs_health = F.softmax(out_health, dim=1)[0]
        max_prob_health, pred_idx_health = torch.max(probs_health, 0)

    pred_health_class = DISEASE_CLASS_NAMES[pred_idx_health.item()]
    response["health_prediction"] = pred_health_class
    response["health_confidence"] = float(max_prob_health.item())

    # STEP 3: Stage2 category & disease lookup
    if pred_health_class == "Unhealthy" and max_prob_health.item() > THRESHOLD_UNHEALTHY:
        with torch.no_grad():
            out_stage2 = stage2_model(img_tensor)
            probs_stage2 = F.softmax(out_stage2, dim=1)[0]
            max_prob_stage2, pred_idx_stage2 = torch.max(probs_stage2, 0)

        category = STAGE2_CLASS_NAMES[pred_idx_stage2.item()]
        response["category"] = category
        response["category_confidence"] = float(max_prob_stage2.item())
        response["possible_diseases"] = CATEGORY_DISEASES.get(category, [])

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
