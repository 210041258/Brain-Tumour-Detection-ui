import gradio as gr
from PIL import Image
import io
import base64
from main_predict import predict_brain_tumor_batch  # updated function for batch


with gr.Blocks() as app: 
    gr.Markdown("## 🧠 Brain Tumor Detection using Vision Transformer (ViT)")
    gr.Markdown("Upload one or more MRI images to detect brain tumors and view images in Full Mode with zoom support.")

    image_input = gr.File(file_types=[".png", ".jpg", ".jpeg"], file_count="multiple", label="Upload MRI Images")
    predict_button = gr.Button("🚀 Predict")

    with gr.Tabs():
        with gr.TabItem("📋 Summary"):
            summary_output = gr.Markdown()
        with gr.TabItem("📄 Detailed Reports"):
            detailed_report_output = gr.Markdown()
        with gr.TabItem("🧬 Tumor Types"):
            tumor_types_output = gr.DataFrame(
                headers=["Image", "Prediction", "Confidence %"],
                datatype=["str", "str", "number"],
                interactive=True
            )
    
    current_predictions = gr.State([])

    # Preview section, hidden by default
    with gr.Row(visible=False) as preview_row:
        preview_image = gr.Image(label="Selected Image")
        preview_text = gr.Markdown()

    def show_image_details(evt: gr.SelectData, predictions):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if idx < len(predictions):
            pred = predictions[idx]
            img = pred["image"]
            
            # Handle different image data types gracefully
            try:
                if hasattr(img, 'read'):  # file-like object
                    img.seek(0)
                    image = Image.open(img).convert("RGB")
                elif isinstance(img, bytes):
                    image = Image.open(io.BytesIO(img)).convert("RGB")
                elif isinstance(img, str):
                    image = Image.open(img).convert("RGB")
                else:
                    return None, "Invalid image data"
            except Exception as e:
                return None, f"Error loading image: {str(e)}"
            
            return (
                image,
                f"""
                ### 🖼️ Image: {pred['filename']}
                ### 🧠 Prediction: **{pred['class'].upper()}**
                ### 🔍 Confidence: **{pred['confidence']:.2f}%**
                """
            )
        return None, "No prediction data available"


    # Prediction click logic
    predict_button.click(
        fn=predict_brain_tumor_batch,
        inputs=image_input,
        outputs=[summary_output, detailed_report_output, tumor_types_output, current_predictions]
    )

    # When selecting a row, show preview
    tumor_types_output.select(
        fn=show_image_details,
        inputs=current_predictions,
        outputs=[preview_image, preview_text]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=preview_row
    )


app.launch()