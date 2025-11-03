import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ==============================
# CONFIG
# ==============================
LEAF_MODEL_PATH = "models/leaf_detector.pth"
DISEASE_MODEL_PATH = "models/best_cpu_model.pth"
SPECIFIC_DISEASE_MODEL_PATH = "models/disease_stage2_best_model.pth"


DEVICE = torch.device("cpu")
LEAF_CLASS_NAMES = ["Leaf", "Not Leaf"]
DISEASE_CLASS_NAMES = ["Dry", "Healthy", "Unhealthy"]

# 41 specific disease classes (as trained)
SPECIFIC_DISEASE_CLASSES = [
     "Anthracnose",
    "Anthrax_Leaf",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Bacterial_Blight",
    "Bacterial_Canker",
    "Bituminous_Leaf",
    "Black_Spot",
    "Cherry_including_sour___Powdery_mildew",
    "Curl_Leaf",
    "Curl_Virus",
    "Cutting_Weevil",
    "Deficiency_Leaf",
    "Die_Back",
    "Entomosporium_Leaf_Spot_on_woody_ornamentals",
    "Felt_Leaf",
    "Fungal_Leaf_Spot",
    "Gall_Midge",
    "Leaf_Blight",
    "Leaf_Gall",
    "Leaf_Holes",
    "Leaf_blight_Litchi_leaf_diseases",
    "Litchi_algal_spot_in_non-direct_sunlight",
    "Litchi_anthracnose_on_cloudy_day",
    "Litchi_leaf_mites_in_direct_sunlight",
    "Litchi_mayetiola_after_raining",
    "Pepper__bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Powdery_Mildew",
    "Sooty_Mould",
    "Spider_Mites",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus"
]

THRESHOLD_LEAF = 0.8
THRESHOLD_UNHEALTHY = 0.6


@st.cache_resource
def load_leaf_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LEAF_CLASS_NAMES))
    model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
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

@st.cache_resource
def load_specific_disease_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(SPECIFIC_DISEASE_CLASSES))
    )
    model.load_state_dict(torch.load(SPECIFIC_DISEASE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

leaf_model = load_leaf_model()
disease_model = load_disease_model()
specific_disease_model = load_specific_disease_model()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


st.set_page_config(page_title=" Leaf Health Detector", page_icon="")
st.title(" Leaf Health Classification")
st.write("Upload an image. The app will check if it's a leaf, predict health, and if unhealthy, show the exact specific disease.")

uploaded_file = st.file_uploader(" Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=" Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

 
    with torch.no_grad():
        outputs = leaf_model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        max_prob, pred_idx = torch.max(probs, 0)

    if LEAF_CLASS_NAMES[pred_idx.item()] == "Not Leaf" and max_prob.item() > THRESHOLD_LEAF:
        st.error(" This image is **not a leaf**. Please upload a valid leaf image.")
    else:
        st.success(" This is a leaf image. Predicting health status...")

       
  
        with torch.no_grad():
            outputs2 = disease_model(img_tensor)
            probs2 = F.softmax(outputs2, dim=1)[0]
            max_prob2, pred_idx2 = torch.max(probs2, 0)

        pred_class = DISEASE_CLASS_NAMES[pred_idx2.item()]
        st.subheader(f" Health Prediction: **{pred_class}**")
        st.progress(float(max_prob2.item()))
        st.caption(f"Confidence: {max_prob2.item():.2%}")

   
        if pred_class == "Unhealthy" and max_prob2.item() > THRESHOLD_UNHEALTHY:
            st.info(" Predicting specific disease...")
            with torch.no_grad():
                outputs3 = specific_disease_model(img_tensor)
                probs3 = F.softmax(outputs3, dim=1)[0]
                max_prob3, pred_idx3 = torch.max(probs3, 0)

            specific_disease = SPECIFIC_DISEASE_CLASSES[pred_idx3.item()]
            st.subheader(f" Specific Disease: **{specific_disease}**")
            st.progress(float(max_prob3.item()))
            st.caption(f"Confidence: {max_prob3.item():.2%}")
