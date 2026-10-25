
import json
import numpy as np
import gradio as gr
import tensorflow as tf
from pathlib import Path

BASE = Path(__file__).parent
MODEL_DIR = BASE / "final_model"

model = tf.keras.models.load_model(
    MODEL_DIR / "asl_signer_independent_lstm.keras"
)

labels = np.load(MODEL_DIR / "label_classes.npy", allow_pickle=False)

with np.load(MODEL_DIR / "preprocessing_scaler.npz") as data:
    scaler_mean = data["mean"].astype(np.float32)
    scaler_scale = data["scale"].astype(np.float32)

def predict_features(text):
    try:
        values = np.fromstring(text.replace(",", " "), sep=" ")

        if values.size != 8 * 258:
            return {
                "Error": (
                    f"Expected 2064 numbers (8 frames × 258 features); "
                    f"received {values.size}."
                )
            }

        if not np.isfinite(values).all():
            return {"Error": "Input contains NaN or infinite values."}

        X = values.reshape(1, 8, 258).astype(np.float32)
        X = (X - scaler_mean) / np.maximum(scaler_scale, 1e-7)

        probabilities = model.predict(X, verbose=0)[0]
        top_indices = np.argsort(probabilities)[-5:][::-1]

        return {
            str(labels[i]): float(probabilities[i])
            for i in top_indices
        }

    except Exception as exc:
        return {"Error": str(exc)}

with gr.Blocks(title="ASL Sign Classifier") as demo:
    gr.Markdown(
        "# ASL Sign Classifier\n"
        "Classifies one pre-extracted sequence of 8 frames and 258 features "
        "into one of 100 trained sign classes.\n\n"
        "**This demo does not accept raw videos or webcam input.**"
    )

    features = gr.Textbox(
        label="Feature values",
        placeholder="Enter 2064 numbers separated by spaces or commas",
        lines=5
    )
    output = gr.Label(label="Top 5 predictions", num_top_classes=5)
    button = gr.Button("Predict")
    button.click(predict_features, inputs=features, outputs=output)

if __name__ == "__main__":
    import os
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=False
    )
    
