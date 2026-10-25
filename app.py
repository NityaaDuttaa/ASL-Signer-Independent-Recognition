import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "final_model"
POSE_TASK = MODEL_DIR / "pose_landmarker.task"
HAND_TASK = MODEL_DIR / "hand_landmarker.task"

SEQUENCE_LENGTH = 8
FEATURES_PER_FRAME = 258
EXPECTED_VALUES = SEQUENCE_LENGTH * FEATURES_PER_FRAME

# Load trained assets once when the server starts.
model = tf.keras.models.load_model(MODEL_DIR / "asl_signer_independent_lstm.keras")
labels = np.load(MODEL_DIR / "label_classes.npy", allow_pickle=False)

with np.load(MODEL_DIR / "preprocessing_scaler.npz") as scaler:
    scaler_mean = scaler["mean"].astype(np.float32)
    scaler_scale = scaler["scale"].astype(np.float32)


def create_landmarkers():
    missing = [str(p.name) for p in (POSE_TASK, HAND_TASK) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MediaPipe task model file(s): "
            + ", ".join(missing)
            + ". Add both files to the final_model folder in your GitHub repository."
        )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_TASK)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_TASK)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_options),
        vision.HandLandmarker.create_from_options(hand_options),
    )


def extract_frame_features(rgb_frame, timestamp_ms, pose_detector, hand_detector):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    pose_result = pose_detector.detect_for_video(image, timestamp_ms)
    hand_result = hand_detector.detect_for_video(image, timestamp_ms)

    # Pose: 33 landmarks x (x, y, z, visibility) = 132 values.
    pose_values = np.zeros(132, dtype=np.float32)
    if pose_result.pose_landmarks:
        values = []
        for point in pose_result.pose_landmarks[0]:
            values.extend([point.x, point.y, point.z, point.visibility])
        pose_values = np.asarray(values, dtype=np.float32)

    # Hands: 2 hands x 21 landmarks x (x, y, z) = 126 values.
    # MediaPipe labels each detected hand as Left or Right. Missing hands stay zero.
    hand_values = {"Left": np.zeros(63, dtype=np.float32),
                   "Right": np.zeros(63, dtype=np.float32)}
    if hand_result.hand_landmarks and hand_result.handedness:
        for landmarks, handedness in zip(
            hand_result.hand_landmarks, hand_result.handedness
        ):
            side = handedness[0].category_name
            if side in hand_values:
                values = []
                for point in landmarks:
                    values.extend([point.x, point.y, point.z])
                hand_values[side] = np.asarray(values, dtype=np.float32)

    # Preserve the project's 258-value layout: pose + left hand + right hand.
    return np.concatenate(
        [pose_values, hand_values["Left"], hand_values["Right"]]
    ).astype(np.float32)


def extract_video_features(video_path):
    if not video_path:
        raise ValueError("Upload a video first.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0:
        cap.release()
        raise ValueError("No readable frames were found in this video.")

    # Sample 8 evenly spaced frames, including the beginning and end.
    frame_indices = np.linspace(0, max(frame_count - 1, 0), SEQUENCE_LENGTH).astype(int)
    pose_detector, hand_detector = create_landmarkers()
    features = []
    try:
        for index, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                features.append(np.zeros(FEATURES_PER_FRAME, dtype=np.float32))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_index / fps) * 1000) if fps and fps > 0 else index * 100
            features.append(
                extract_frame_features(rgb, timestamp_ms, pose_detector, hand_detector)
            )
    finally:
        cap.release()
        pose_detector.close()
        hand_detector.close()

    sequence = np.stack(features).astype(np.float32)
    if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise ValueError(f"Unexpected extracted feature shape: {sequence.shape}")
    return sequence


def predict_sequence(sequence):
    X = sequence.reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME).astype(np.float32)
    X = (X - scaler_mean) / np.maximum(scaler_scale, 1e-7)
    probabilities = model.predict(X, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-5:][::-1]
    results = {
        str(labels[i]): float(probabilities[i])
        for i in top_indices
    }
    best_label = str(labels[int(np.argmax(probabilities))])
    best_score = float(np.max(probabilities))
    return best_label, best_score, results


def predict_video(video_path):
    try:
        sequence = extract_video_features(video_path)
        best_label, best_score, results = predict_sequence(sequence)
        return (
            f"Predicted sign: {best_label}\nModel score: {best_score:.1%}\n\n"
            "This is a research prototype. Scores are model outputs, not guaranteed probabilities.",
            results,
        )
    except Exception as exc:
        return f"Could not process this video: {exc}", {}


def predict_features(text):
    try:
        values = np.fromstring((text or "").replace(",", " "), sep=" ")
        if values.size != EXPECTED_VALUES:
            return {
                "Error": (
                    f"Expected {EXPECTED_VALUES} numbers "
                    f"({SEQUENCE_LENGTH} frames × {FEATURES_PER_FRAME} features); "
                    f"received {values.size}."
                )
            }
        if not np.isfinite(values).all():
            return {"Error": "Input contains NaN or infinite values."}
        _, _, results = predict_sequence(values.reshape(SEQUENCE_LENGTH, FEATURES_PER_FRAME))
        return results
    except Exception as exc:
        return {"Error": str(exc)}


custom_css = """
.gradio-container {max-width: 1100px !important; margin: auto !important;}
.hero {padding: 24px; border-radius: 18px; background: linear-gradient(120deg,#111827,#243b53); color: white;}
.hero h1 {margin-bottom: 8px; font-size: 2rem;}
.hero p {opacity: .88; margin-bottom: 0;}
.note {font-size: .92rem; opacity: .8;}
"""

with gr.Blocks(title="ASL Signer Independent Recognition", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    <div class="hero">
      <h1>🤟 ASL Signer Independent Recognition</h1>
      <p>Upload a short sign video and run it through the trained sequence model.</p>
    </div>
    """)
    gr.Markdown(
        "The prototype samples **8 frames**, extracts pose and hand landmarks, "
        "and predicts one of the model's trained sign classes."
    )

    with gr.Tab("Recognize a sign from video"):
        video = gr.Video(label="Upload ASL video", sources=["upload"], format="mp4")
        run_button = gr.Button("Analyze sign", variant="primary")
        prediction = gr.Textbox(label="Main prediction", lines=3, interactive=False)
        top_predictions = gr.Label(label="Top 5 model outputs", num_top_classes=5)
        run_button.click(predict_video, inputs=video, outputs=[prediction, top_predictions])

    with gr.Tab("Advanced: pre-extracted features"):
        gr.Markdown(
            "For debugging only. Enter exactly 2,064 numeric values: "
            "8 frames × 258 features per frame."
        )
        feature_text = gr.Textbox(
            label="Feature values",
            placeholder="Enter numbers separated by spaces or commas",
            lines=5,
        )
        feature_button = gr.Button("Predict from features")
        feature_output = gr.Label(label="Top 5 predictions", num_top_classes=5)
        feature_button.click(predict_features, inputs=feature_text, outputs=feature_output)

    gr.Markdown(
        "**Research limitation:** this is an experimental classifier, not a reliable "
        "general-purpose sign-language translator. Performance depends on lighting, "
        "camera angle, signer variation, and the training data."
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=False,
    )
