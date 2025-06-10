import os
import torch
import urllib.request
import random  # Added for the random noise transform
from torchvision import transforms 
from PIL import Image
import torchvision.models as models
from typing import List, Tuple


MODEL_PATH = "C:\\Users\\asdal\\Downloads\\Brain-Tumour-Detection-main\\Brain-Tumour-Detection-main\\UI-AHMED-\\vit_brain_tumor.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1UaFw3vAuimY6r47mYbkyFXmtsjwDjsoP"  # Updated Google Drive download link



# ========== DOWNLOAD MODEL IF NEEDED ========== #
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Model downloaded.")
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            raise

def load_model(model_path: str = MODEL_PATH, device: str = "cpu") -> torch.nn.Module:
    """
    Load the brain tumor classification model with proper error handling
    """
    try:
        # 1. Initialize the base ViT model
        model = models.vit_b_16(weights=None)  # We'll load our own weights
        
        # 2. Modify model head for our 4-class problem
        model.heads.head = torch.nn.Sequential(
            torch.nn.Linear(model.heads.head.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 4)  # 4 output classes
        )
        
        # 3. Load our fine-tuned weights
        if not os.path.exists(model_path):
            download_model()
            
        state_dict = torch.load(model_path, map_location=torch.device(device))
        
        # Handle potential mismatch in layer names
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):  # Handle DDP/DP saved models
                k = k[7:]
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        # 4. Configure for inference
        model.to(device)
        model.eval()
        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
            
        return model
        
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}") from e

# ======= TRANSFORM PIPELINE ======= #
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01 if random.random() < 0.3 else x)
])

# ======= PREDICTION FUNCTION ======= #
def predict_brain_tumor_batch(img_list: list) -> Tuple[str, str, List[List], List[dict]]:
    results = []
    detailed_reports = []
    tumor_types_data = []    # For dataframe: list of rows [image, prediction, confidence %]
    current_predictions = [] # For state, detailed info per image
    
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    class_descriptions = {
        'glioma': 'A type of tumor that occurs in the brain and spinal cord, arising from glial cells.',
        'meningioma': 'A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.',
        'notumor': 'No tumor detected in the brain MRI scan.',
        'pituitary': 'A tumor affecting the pituitary gland at the base of the brain that controls many hormonal functions.'
    }
    
    if not img_list:
        return ("⚠️ No images uploaded", "Please upload MRI images for analysis", [], [])
    
    for idx, img_data in enumerate(img_list, start=1):
        try:
            # Handle input types
            if isinstance(img_data, str):
                img = Image.open(img_data).convert("RGB")
                filename = img_data.split("/")[-1]
            elif hasattr(img_data, 'read'):
                img = Image.open(img_data).convert("RGB")
                filename = getattr(img_data, 'name', f"image_{idx}")
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                filename = f"image_{idx}"
            else:
                raise ValueError(f"Unsupported image type: {type(img_data)}")
            
            # Model prediction
            input_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                prediction = torch.argmax(output, dim=1).item()
            
            predicted_class = class_names[prediction]
            confidence_score = probabilities[prediction].item()
            
            # Summary text
            results.append(f"{idx}. 🖼️ **{filename}** → 🧠 *{predicted_class.title()}* ({confidence_score*100:.1f}%)")
            
            # Detailed report (same as before)
            detail_report = [
                f"📄 COMPREHENSIVE ANALYSIS REPORT: {filename}",
                "="*50,
                f"🏆 FINAL PREDICTION: {predicted_class.upper()} ({confidence_score*100:.2f}% confidence)",
                "",
                f"📖 Description: {class_descriptions[predicted_class]}",
                "",
                "📊 CLASS PROBABILITY DISTRIBUTION:",
            ]
            max_len = 20
            for i, class_name in enumerate(class_names):
                percentage = probabilities[i].item() * 100
                bar_len = int(percentage / 100 * max_len)
                bar = '█' * bar_len + ' ' * (max_len - bar_len)
                highlight = "→" if i == prediction else " "
                detail_report.append(
                    f"  {highlight} {class_name.title():<12} {percentage:5.2f}% |{bar}| {probabilities[i].item():.4f}"
                )
            # Add analysis metrics (assuming your helper functions exist)
            detail_report.extend([
                "",
                "🔍 CONFIDENCE ANALYSIS:",
                f"- Prediction Confidence Score: {confidence_score:.4f}",
                f"- Confidence Level: {get_confidence_level(confidence_score)}",
                f"- Second Most Likely Class: {get_second_most_likely(probabilities, class_names)}",
                "",
                "⚖️ PREDICTION RELIABILITY INDICATORS:",
                f"- Probability Spread: {calculate_probability_spread(probabilities):.3f} (higher is better)",
                f"- Uncertainty Index: {calculate_uncertainty(probabilities):.3f} (lower is better)",
                "",
                "💡 CLINICAL CONSIDERATIONS:",
                get_clinical_considerations(predicted_class, confidence_score)
            ])
            detailed_reports.append("\n".join(detail_report))
            
            # Add row for dataframe output
            tumor_types_data.append([filename, predicted_class.title(), round(confidence_score * 100, 2)])
            
            # Add full prediction data for state (used in preview etc.)
            current_predictions.append({
                "filename": filename,
                "class": predicted_class,
                "confidence": confidence_score * 100,
                "image": img_data
            })
            
        except Exception as e:
            error_msg = f"❌ Error processing image {idx}: {str(e)}"
            results.append(error_msg)
            detailed_reports.append(error_msg)
            tumor_types_data.append([f"image_{idx}", "error", 0])
            current_predictions.append({
                "filename": f"image_{idx}",
                "class": "error",
                "confidence": 0,
                "image": None
            })
    
    return (
        "\n".join(results) if results else "No results generated",
        "\n\n".join(detailed_reports) if detailed_reports else "No detailed reports generated",
        tumor_types_data,
        current_predictions
    )


# Helper functions
def get_confidence_level(score: float) -> str:
    if score > 0.95: return "⭐⭐⭐⭐⭐ Exceptional (>95%)"
    elif score > 0.9: return "⭐⭐⭐⭐ Very High (90-95%)"
    elif score > 0.75: return "⭐⭐⭐ High (75-90%)"
    elif score > 0.6: return "⭐⭐ Moderate (60-75%)"
    elif score > 0.4: return "⭐ Low (40-60%)"
    else: return "❓ Very Low (<40%) - Consider manual review"

def get_second_most_likely(probs, classes) -> str:
    sorted_probs = sorted(zip(probs, classes), reverse=True)
    return f"{sorted_probs[1][1].title()} ({sorted_probs[1][0].item()*100:.2f}%)"

def calculate_probability_spread(probs) -> float:
    sorted_probs = torch.sort(probs, descending=True).values
    return sorted_probs[0] - sorted_probs[1]

def calculate_uncertainty(probs) -> float:
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()  # Added small epsilon to avoid log(0)

def get_clinical_considerations(pred_class, confidence) -> str:
    considerations = {
        'glioma': [
            "Gliomas can be aggressive and require prompt attention",
            "Recommend follow-up with neurologist and MRI spectroscopy",
            "Consider grading evaluation (low-grade vs high-grade)"
        ],
        'meningioma': [
            "Most meningiomas are benign (WHO Grade I)",
            "Recommend monitoring growth rate if asymptomatic",
            "Surgical resection may be indicated for symptomatic cases"
        ],
        'notumor': [
            "No immediate intervention needed",
            "Recommend routine follow-up if clinically indicated",
            "Consider alternative diagnoses if symptoms persist"
        ],
        'pituitary': [
            "Endocrine evaluation recommended",
            "Assess for hormonal hypersecretion syndromes",
            "Monitor for visual field defects if macroadenoma"
        ]
    }
    
    base = "\n".join([f"- {item}" for item in considerations[pred_class]])
    if confidence < 0.7:
        base += "\n\n⚠️ NOTE: Due to lower confidence in prediction, consider:"
        base += "\n- Additional imaging (contrast-enhanced MRI)"
        base += "\n- Second opinion from neuroradiologist"
        base += "\n- Clinical correlation with patient symptoms"
    return base



# Initialize model
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise