import gradio as gr
from PIL import Image
import io
from main_predict import predict_brain_tumor_batch  # updated function for batch








CSS = """
html,
body, .gradio-container {
    background: linear-gradient(120deg, #e0e7ff 0%, #f8fafc 100%);
    min-height: 100vh;
    font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    color: #232946;
    scroll-behavior: smooth;
}

::-webkit-scrollbar {
    width: 10px;
    background: #e0e7ff;
    border-radius: 8px;
}
::-webkit-scrollbar-thumb {
    background: #a5b4fc;
    border-radius: 8px;
}

.main-section {
    padding: 40px 0 24px 0;
    background: linear-gradient(120deg, #f1f5f9 60%, #c7d2fe 100%);
    border-radius: 22px;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(60,72,88,0.15);
    border: 2.5px solid #818cf8;
    transition: box-shadow 0.3s, background 0.3s, color 0.3s;
    animation: fadeIn 1.2s;
}
.main-section:hover {
    box-shadow: 0 16px 40px rgba(60,72,88,0.22);
    background: linear-gradient(120deg, #818cf8 60%, #6366f1 100%);
    color: #fff;
}

.section-header {
    font-size: 1.5em;
    font-weight: 900;
    color: #3730a3;
    margin-top: 36px;
    margin-bottom: 18px;
    letter-spacing: 0.04em;
    text-shadow: 0 2px 0 #e0e7ff;
    border-left: 6px solid #6366f1;
    padding-left: 16px;
    background: linear-gradient(90deg, #f3f4f6 80%, #e0e7ff 100%);
    border-radius: 8px;
    box-shadow: 0 2px 8px #c7d2fe;
}

.gr-button, .gr-button-primary {
    background: linear-gradient(90deg, #6366f1 60%, #818cf8 100%);
    color: #fff;
    font-weight: 800;
    border-radius: 12px;
    border: none;
    box-shadow: 0 2px 12px #a5b4fc;
    transition: background 0.2s, box-shadow 0.2s, transform 0.1s, color 0.2s;
    font-size: 1.08em;
    padding: 12px 28px;
    letter-spacing: 0.03em;
}
.gr-button:hover, .gr-button-primary:hover {
    background: linear-gradient(90deg, #232946 60%, #6366f1 100%);
    color: #ffe066;
    box-shadow: 0 6px 20px #6366f1;
    transform: translateY(-2px) scale(1.04);
}

.gr-tabs, .gr-tabitem {
    background: linear-gradient(90deg, #f1f5f9 80%, #e0e7ff 100%);
    border-radius: 14px;
    margin-bottom: 14px;
    box-shadow: 0 2px 10px #c7d2fe;
    font-size: 1.07em;
    transition: background 0.2s, color 0.2s;
}
.gr-tabs:hover, .gr-tabitem:hover {
    background: linear-gradient(90deg, #6366f1 60%, #818cf8 100%);
    color: #fff;
}

.confidence-high {
    color: #16a34a;
    font-weight: 800;
    background: linear-gradient(90deg, #dcfce7 60%, #bbf7d0 100%);
    padding: 6px 20px;
    border-radius: 12px;
    font-size: 1.18em;
    border: 2.5px solid #22c55e;
    box-shadow: 0 2px 10px #bbf7d0;
    transition: background 0.2s, color 0.2s;
    letter-spacing: 0.01em;
}
.confidence-high:hover {
    background: linear-gradient(90deg, #bbf7d0 60%, #dcfce7 100%);
    color: #065f46;
}

.confidence-low {
    color: #d97706;
    font-weight: 800;
    background: linear-gradient(90deg, #fef9c3 60%, #fde68a 100%);
    padding: 6px 20px;
    border-radius: 12px;
    font-size: 1.18em;
    border: 2.5px solid #fbbf24;
    box-shadow: 0 2px 10px #fde68a;
    transition: background 0.2s, color 0.2s;
    letter-spacing: 0.01em;
}
.confidence-low:hover {
    background: linear-gradient(90deg, #fde68a 60%, #fef9c3 100%);
    color: #b45309;
}

.warning-icon {
    color: #d97706;
    font-size: 1.28em;
    margin-right: 10px;
    vertical-align: middle;
    filter: drop-shadow(0 2px 2px #fde68a);
}

.prediction-bar {
    height: 28px;
    border-radius: 14px;
    margin-bottom: 12px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1), background 0.2s;
    box-shadow: 0 2px 12px rgba(60,72,88,0.13);
    cursor: pointer;
    background: linear-gradient(90deg, #818cf8 60%, #6366f1 100%);
    position: relative;
    overflow: hidden;
}
.prediction-bar:hover {
    filter: brightness(1.18) drop-shadow(0 2px 12px #a5b4fc);
    border: 2.5px solid #6366f1;
}

.bar-label {
    margin-left: 22px;
    font-weight: 900;
    color: #3730a3;
    font-size: 1.13em;
    letter-spacing: 0.01em;
    text-shadow: 0 1px 0 #e0e7ff;
    transition: color 0.2s;
}

.bar-container {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    background: linear-gradient(90deg, #f1f5f9 60%, #e0e7ff 100%);
    border-radius: 14px;
    padding: 7px 20px;
    border: 2.5px solid #c7d2fe;
    box-shadow: 0 2px 10px #e0e7ff;
    transition: background 0.2s, box-shadow 0.2s;
}
.bar-container:hover {
    background: linear-gradient(90deg, #e0e7ff 60%, #f1f5f9 100%);
    box-shadow: 0 6px 20px #a5b4fc;
}

.gr-file, .gr-image {
    border-radius: 12px !important;
    box-shadow: 0 2px 12px #c7d2fe;
    border: 2px solid #a5b4fc;
    background: #f8fafc;
    transition: box-shadow 0.2s, border 0.2s;
}
.gr-file:hover, .gr-image:hover {
    box-shadow: 0 6px 20px #818cf8;
    border: 2.5px solid #6366f1;
}

.gr-markdown {
    font-size: 1.09em;
    line-height: 1.7;
    color: #232946;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(30px);}
    to { opacity: 1; transform: translateY(0);}
}
"""


with gr.Blocks(css=CSS) as app:
    gr.Markdown(
        """
        <div style="display: flex; align-items: center; gap: 18px;">
            <img src="https://img.icons8.com/color/96/brain.png" width="64" style="border-radius: 12px; box-shadow: 0 2px 12px #a5b4fc;">
            <div>
                <h1 style="margin-bottom: 0; font-size: 2.2em; font-weight: 900; color: #3730a3;">Brain Tumor Detection</h1>
                <div style="font-size: 1.15em; color: #6366f1; font-weight: 600;">Powered by Vision Transformer (ViT)</div>
            </div>
        </div>
        """,
        elem_id="header"
    )
    gr.Markdown(
        """
        <div style="margin-top: 18px; font-size: 1.13em;">
            Upload <b>one or more MRI images</b> to detect brain tumors.<br>
            <span style="color:#818cf8;">Enjoy a modern, interactive experience with zoomable previews and detailed reports.</span>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.File(
                file_types=[".png", ".jpg", ".jpeg"],
                file_count="multiple",
                label="📤 Upload MRI Images",
                elem_id="image-upload"
            )
        with gr.Column(scale=1, min_width=180):
            predict_button = gr.Button("🚀 Predict", elem_id="predict-btn")
            clear_button = gr.Button("🧹 Clear All", elem_id="clear-btn")

            

    with gr.Tabs():
        with gr.TabItem("📋 Summary"):
            summary_output = gr.HTML(elem_id="summary-md")
        with gr.TabItem("📄 Detailed Reports"):
            detailed_report_output = gr.HTML(elem_id="detailed-html")
        with gr.TabItem("🧬 Tumor Types"):
            tumor_types_output = gr.Dataframe(
                headers=["Image", "Prediction", "Confidence %"],
                datatype=["str", "str", "number"],
                interactive=True,
                elem_id="tumor-types-df",
                wrap=True,
                row_count=5,
                col_count=(3, "fixed"),
                visible=True,
                label="Tumor Type Predictions"
            )

    current_predictions = gr.State([])

    # Preview section, hidden by default
    with gr.Row(visible=False) as preview_row:
        with gr.Column(scale=1):
            preview_image = gr.Image(label="Selected Image", show_label=True, elem_id="preview-img")
        with gr.Column(scale=2):
            preview_text = gr.Markdown(elem_id="preview-md")

    def clear_all():
        return (
            None,  # summary_output
            None,  # detailed_report_output
            gr.update(value=None),  # tumor_types_output
            [],    # current_predictions
            None,  # preview_image
            "",    # preview_text
            None   # image_input
        )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            summary_output,
            detailed_report_output,
            tumor_types_output,
            current_predictions,
            preview_image,
            preview_text,
            image_input
        ]
    )

    def show_image_details(evt: gr.SelectData, predictions):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if idx < len(predictions):
            pred = predictions[idx]
            img = pred["image"]
            try:
                if hasattr(img, 'read'):
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
                <div style="font-size:1.18em;">
                    <b>🖼️ Image:</b> {pred['filename']}<br>
                    <b>🧠 Prediction:</b> <span style="color:#6366f1;">{pred['class'].upper()}</span><br>
                    <b>🔍 Confidence:</b> <span style="color:#16a34a;">{pred['confidence']:.2f}%</span>
                </div>
                """
            )
        return None, "No prediction data available"

    predict_button.click(
        fn=predict_brain_tumor_batch,
        inputs=image_input,
        outputs=[summary_output, detailed_report_output, tumor_types_output, current_predictions]
    )

    tumor_types_output.select(
        fn=show_image_details,
        inputs=current_predictions,
        outputs=[preview_image, preview_text]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=preview_row
    )


app.launch()