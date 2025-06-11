import os
import torch
import urllib.request
import random  # Added for the random noise transform
from torchvision import transforms 
from PIL import Image
import torchvision.models as models
from typing import List, Tuple
import io


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
from typing import Tuple, List
from PIL import Image
import io
import torch

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

    # Improved CSS styles for HTML output (fixed for better compatibility)
    css_styles = """
    <style type="text/css">
    .report-container { font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; border-radius: 12px; padding: 24px; margin-bottom: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .report-title { font-size: 1.3em; font-weight: bold; color: #2d3748; margin-bottom: 8px; }
    .divider { border-bottom: 2px solid #e2e8f0; margin: 12px 0 18px 0; }
    .prediction-main { font-size: 1.1em; color: #2563eb; font-weight: 600; margin-bottom: 8px; }
    .desc { color: #374151; margin-bottom: 10px; }
    .prob-table { width: 100%; border-collapse: collapse; margin-bottom: 10px; }
    .prob-table th, .prob-table td { padding: 6px 10px; text-align: left; border: none; }
    .prob-table th { background: #e0e7ef; color: #22223b; }
    .prob-bar, .prob-bar-main { display: inline-block; height: 16px; border-radius: 6px; vertical-align: middle; }
    .prob-bar { background: #c7d2fe; }
    .prob-bar-main { background: #6366f1; }
    .note { color: #b91c1c; font-weight: 500; margin: 8px 0; }
    .metrics { background: #f1f5f9; border-radius: 8px; padding: 10px 14px; margin: 10px 0; }
    .clinical { background: #fef9c3; border-radius: 8px; padding: 10px 14px; color: #92400e; margin: 10px 0; }
    .filename { color: #64748b; font-size: 0.98em; }
    </style>
    """

    if not img_list:
        return ("⚠️ No images uploaded", "Please upload MRI images for analysis", [], [])

    for idx, img_data in enumerate(img_list, start=1):
        try:
            # Load image
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

            # Perform 1 round with 10 predictions each
            rounds = 20
            preds_per_round = 20
            round_confidences = []

            for r in range(rounds):
                class_conf_sum = torch.zeros(len(class_names))
                for _ in range(preds_per_round):
                    input_tensor = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    class_conf_sum += probabilities
                avg_confidence = class_conf_sum / preds_per_round
                round_confidences.append(avg_confidence)

            # Pick best round based on highest max confidence
            best_round_idx = max(range(rounds), key=lambda i: round_confidences[i].max().item())
            best_avg_conf = round_confidences[best_round_idx]

            # Determine top 2 classes
            sorted_confidences = torch.topk(best_avg_conf, 2)
            top1_idx, top2_idx = sorted_confidences.indices.tolist()
            top1_conf, top2_conf = sorted_confidences.values.tolist()

            predicted_class = class_names[top1_idx]
            confidence_score = top1_conf
            second_class = class_names[top2_idx]
            second_confidence = top2_conf

            # Check if the difference between top 2 is ≤ 15%
            diff_percent = (top1_conf - top2_conf) * 100
            show_dual_prediction = diff_percent <= 15.0

            # Summary output (plain text)
            # Improved summary output with enhanced CSS styling
            if show_dual_prediction:
                results.append(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, #fef9c3 60%, #f1f5f9 100%);
                        border-radius: 10px;
                        padding: 16px 22px;
                        margin: 12px 0 18px 0;
                        box-shadow: 0 2px 8px rgba(99,102,241,0.07);
                        border-left: 6px solid #f59e42;
                        font-family: 'Segoe UI', Arial, sans-serif;
                        font-size: 1.08em;
                        color: #92400e;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    ">
                        <span style="font-size:1.3em;">🖼️</span>
                        <span class="filename" style="color:#64748b;font-style:italic;">{filename}</span>
                        <span style="margin-left:8px;">
                            <b style="color:#b45309;">🧠 {predicted_class.title()} <span style="font-weight:400;">or</span> {second_class.title()}</b>
                            <span style="color:#b45309;">({confidence_score*100:.1f}% vs {second_confidence*100:.1f}%)</span>
                            <span style="background:#fde68a;color:#b45309;padding:2px 8px;border-radius:6px;font-size:0.98em;margin-left:8px;">≤15% diff</span>
                        </span>
                    </div>
                    """
                )
            else:
                results.append(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, #e0e7ef 60%, #f1f5f9 100%);
                        border-radius: 10px;
                        padding: 16px 22px;
                        margin: 12px 0 18px 0;
                        box-shadow: 0 2px 8px rgba(99,102,241,0.07);
                        border-left: 6px solid #6366f1;
                        font-family: 'Segoe UI', Arial, sans-serif;
                        font-size: 1.08em;
                        color: #1e293b;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    ">
                        <span style="font-size:1.3em;">🖼️</span>
                        <span class="filename" style="color:#64748b;font-style:italic;">{filename}</span>
                        <span style="margin-left:15px;">
                            <b style="color:#2563eb;">🧠 {predicted_class.title()}</b>
                            <span style="color:#2563eb;">({confidence_score*100:.1f}%)</span>
                            <span style="background:#c7d2fe;color:#3730a3;padding:2px 8px;border-radius:6px;font-size:0.98em;margin-left:8px;">Best of 20 rounds</span>
                        </span>
                    </div>
                    """
                )

            # Improved CSS styles for HTML output (modern, clean, accessible)
            css_styles = """
            <style type="text/css">
            .report-container {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: #f4f8fb;
                border-radius: 14px;
                padding: 28px 28px 22px 28px;
                margin-bottom: 36px;
                box-shadow: 0 4px 16px rgba(30, 41, 59, 0.08);
                border: 1px solid #e2e8f0;
                max-width: 650px;
            }
            .report-title {
                font-size: 1.35em;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 10px;
                letter-spacing: 0.01em;
            }
            .divider {
                border-bottom: 2px solid #e0e7ef;
                margin: 14px 0 20px 0;
            }
            .prediction-main {
                font-size: 1.13em;
                color: #2563eb;
                font-weight: 600;
                margin-bottom: 10px;
                letter-spacing: 0.01em;
            }
            .desc {
                color: #334155;
                margin-bottom: 12px;
                font-size: 1.01em;
            }
            .prob-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0 4px;
                margin-bottom: 12px;
            }
            .prob-table th, .prob-table td {
                padding: 7px 12px;
                text-align: left;
                border: none;
                font-size: 0.99em;
            }
            .prob-table th {
                background: #e0e7ef;
                color: #22223b;
                font-weight: 600;
            }
            .prob-bar, .prob-bar-main {
                display: inline-block;
                height: 18px;
                border-radius: 7px;
                vertical-align: middle;
                transition: width 0.4s cubic-bezier(.4,2.3,.3,1);
            }
            .prob-bar {
                background: linear-gradient(90deg, #c7d2fe 60%, #e0e7ef 100%);
            }
            .prob-bar-main {
                background: linear-gradient(90deg, #6366f1 70%, #818cf8 100%);
            }
            .note {
                color: #b91c1c;
                font-weight: 500;
                margin: 10px 0 8px 0;
                font-size: 1.01em;
            }
            .metrics {
                background: #f1f5f9;
                border-radius: 9px;
                padding: 12px 16px;
                margin: 12px 0;
                font-size: 0.98em;
                color: #334155;
                border-left: 4px solid #6366f1;
            }
            .clinical {
                background: #fef9c3;
                border-radius: 9px;
                padding: 12px 16px;
                color: #92400e;
                margin: 12px 0 0 0;
                font-size: 0.99em;
                border-left: 4px solid #f59e42;
            }
            .filename {
                color: #64748b;
                font-size: 0.99em;
                font-style: italic;
                letter-spacing: 0.01em;
            }
            @media (max-width: 700px) {
                .report-container { padding: 16px 6vw; }
                .prob-table th, .prob-table td { padding: 6px 4vw; }
            }
            </style>
            """

            # Detailed HTML report
            detail_report = [css_styles]
            detail_report.append(f"<div class='report-container'>")
            detail_report.append(f"<div class='report-title'>📄 Comprehensive Analysis Report: <span class='filename'>{filename}</span></div>")
            detail_report.append(f"<div class='divider'></div>")
            detail_report.append(
                f"<div class='prediction-main'>🏆 Final Prediction: <b>{predicted_class.upper()}</b> "
                f"({confidence_score*100:.2f}% confidence) <span style='font-size:0.95em;color:#64748b;'>(Best of 20 rounds)</span></div>"
            )
            detail_report.append(f"<div class='filename'>🏅 Best Round: {best_round_idx+1} / {rounds}</div>")
            detail_report.append(f"<div class='desc'><b>📖 Description:</b> {class_descriptions[predicted_class]}</div>")

            # Probability table
            detail_report.append("<table class='prob-table'><tr><th>Class</th><th>Probability</th><th>Bar</th></tr>")
            max_len = 180
            for i, class_name in enumerate(class_names):
                percentage = best_avg_conf[i].item() * 100
                bar_len = int(percentage / 100 * max_len)
                bar_class = "prob-bar-main" if i == top1_idx else "prob-bar"
                detail_report.append(
                    f"<tr><td>{'→' if i == top1_idx else ''} {class_name.title()}</td>"
                    f"<td>{percentage:5.2f}%</td>"
                    f"<td><div class='{bar_class}' style='width:{bar_len}px'></div></td></tr>"
                )
            detail_report.append("</table>")

            # Optional note on closeness
            if show_dual_prediction:
                detail_report.append(
                    f"<div class='note'>⚠️ The second most likely class is within 15% confidence margin.<br>"
                    f"- Alternative Diagnosis: <b>{second_class.title()}</b> ({second_confidence*100:.2f}%)</div>"
                )

            # Analysis metrics
            detail_report.append("<div class='metrics'>")
            detail_report.append(f"<b>🔍 Confidence Analysis:</b><br>")
            detail_report.append(f"- Prediction Confidence Score: <b>{confidence_score:.4f}</b><br>")
            detail_report.append(f"- Confidence Level: {get_confidence_level(confidence_score)}<br>")
            detail_report.append(f"- Second Most Likely Class: {get_second_most_likely(best_avg_conf, class_names)}<br>")
            detail_report.append("<br><b>⚖️ Prediction Reliability Indicators:</b><br>")
            detail_report.append(f"- Probability Spread: <b>{calculate_probability_spread(best_avg_conf):.3f}</b> (higher is better)<br>")
            detail_report.append(f"- Uncertainty Index: <b>{calculate_uncertainty(best_avg_conf):.3f}</b> (lower is better)")
            detail_report.append("</div>")

            # Clinical considerations
            detail_report.append("<div class='clinical'>")
            detail_report.append(f"<b>💡 Clinical Considerations:</b><br>")
            detail_report.append(get_clinical_considerations(predicted_class, confidence_score).replace('\n', '<br>'))
            detail_report.append("</div>")

            detail_report.append("</div>")  # Close report-container
            detailed_reports.append("".join(detail_report))

            # Dataframe + preview state
            tumor_types_data.append([filename, predicted_class.title(), round(confidence_score * 100, 2)])
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
        "<br>".join(results) if results else "No results generated",
        "<br><br>".join(detailed_reports) if detailed_reports else "No detailed reports generated",
        tumor_types_data,
        current_predictions
    )


# Helper functions
def get_confidence_level(score: float) -> str:
    # Confidence level with structured, clear output
    if score > 0.95:
        return (
            "⭐⭐⭐⭐⭐ Exceptional Confidence (>95%)\n"
            "- The model is extremely certain about this prediction."
        )
    elif score > 0.9:
        return (
            "⭐⭐⭐⭐ Very High Confidence (90-95%)\n"
            "- The prediction is highly reliable."
        )
    elif score > 0.75:
        return (
            "⭐⭐⭐ High Confidence (75-90%)\n"
            "- The model is confident, but clinical review is still advised."
        )
    elif score > 0.6:
        return (
            "⭐⭐ Moderate Confidence (60-75%)\n"
            "- The result is moderately reliable. Consider additional review."
        )
    elif score > 0.4:
        return (
            "⭐ Low Confidence (40-60%)\n"
            "- The prediction is uncertain. Manual review is recommended."
        )
    else:
        return (
            "❓ Very Low Confidence (<40%)\n"
            "- The model is unsure. Strongly consider manual review and further diagnostics."
        )

def get_second_most_likely(probs, classes) -> str:
    # Sort by probability, descending
    sorted_probs = sorted(zip(probs, classes), key=lambda x: x[0], reverse=True)
    return (
        f"Second Most Likely: {sorted_probs[1][1].title()} "
        f"({sorted_probs[1][0].item()*100:.2f}%)"
    )

def calculate_probability_spread(probs) -> float:
    sorted_probs = torch.sort(probs, descending=True).values
    return (sorted_probs[0] - sorted_probs[1]).item()

def calculate_uncertainty(probs) -> float:
    # Entropy: -sum(p*log(p)), add epsilon to avoid log(0)
    return float(-torch.sum(probs * torch.log(probs + 1e-10)).item())

def get_clinical_considerations(pred_class, confidence) -> str:
    considerations = {
        'glioma': [
            "• Gliomas can be aggressive and require prompt attention.",
            "• Recommend follow-up with neurologist and MRI spectroscopy.",
            "• Consider grading evaluation (low-grade vs high-grade)."
        ],
        'meningioma': [
            "• Most meningiomas are benign (WHO Grade I).",
            "• Recommend monitoring growth rate if asymptomatic.",
            "• Surgical resection may be indicated for symptomatic cases."
        ],
        'notumor': [
            "• No immediate intervention needed.",
            "• Recommend routine follow-up if clinically indicated.",
            "• Consider alternative diagnoses if symptoms persist."
        ],
        'pituitary': [
            "• Endocrine evaluation recommended.",
            "• Assess for hormonal hypersecretion syndromes.",
            "• Monitor for visual field defects if macroadenoma."
        ]
    }
    base = "CLINICAL CONSIDERATIONS:\n" + "\n".join(considerations.get(pred_class, []))
    if confidence < 0.7:
        base += (
            "\n\n⚠️ NOTE: Due to lower confidence in prediction, consider:\n"
            "• Additional imaging (contrast-enhanced MRI).\n"
            "• Second opinion from neuroradiologist.\n"
            "• Clinical correlation with patient symptoms."
        )
    return base


# Initialize model
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise
