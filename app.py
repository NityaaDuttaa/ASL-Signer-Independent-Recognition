
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
MAX_WIDTH = 640

model = tf.keras.models.load_model(
    MODEL_DIR / "asl_signer_independent_lstm.keras"
)
labels = np.load(
    MODEL_DIR / "label_classes.npy",
    allow_pickle=False
)

with np.load(MODEL_DIR / "preprocessing_scaler.npz") as scaler:
    scaler_mean = scaler["mean"].astype(np.float32)
    scaler_scale = scaler["scale"].astype(np.float32)
    scaler_scale = np.maximum(scaler_scale, 1e-7)


def create_landmarkers():
    for path in (POSE_TASK, HAND_TASK):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required MediaPipe model: {path.name}"
            )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(POSE_TASK)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(HAND_TASK)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    return (
        vision.PoseLandmarker.create_from_options(pose_options),
        vision.HandLandmarker.create_from_options(hand_options),
    )


def extract_frame_features(
    rgb_frame, timestamp_ms, pose_detector, hand_detector
):
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    pose_result = pose_detector.detect_for_video(
        image, timestamp_ms
    )
    hand_result = hand_detector.detect_for_video(
        image, timestamp_ms
    )

    pose_values = np.zeros(132, dtype=np.float32)

    if pose_result.pose_landmarks:
        values = []
        for point in pose_result.pose_landmarks[0]:
            values.extend([
                point.x, point.y, point.z, point.visibility
            ])
        pose_values = np.asarray(values, dtype=np.float32)

    hand_values = {
        "Left": np.zeros(63, dtype=np.float32),
        "Right": np.zeros(63, dtype=np.float32),
    }

    if hand_result.hand_landmarks and hand_result.handedness:
        for landmarks, handedness in zip(
            hand_result.hand_landmarks,
            hand_result.handedness
        ):
            side = handedness[0].category_name

            if side in hand_values:
                values = []
                for point in landmarks:
                    values.extend([point.x, point.y, point.z])

                hand_values[side] = np.asarray(
                    values, dtype=np.float32
                )

    return np.concatenate([
        pose_values,
        hand_values["Left"],
        hand_values["Right"],
    ]).astype(np.float32)


def extract_video_features(video_path):
    if not video_path:
        raise ValueError("Upload a video first.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count <= 0:
        cap.release()
        raise ValueError("No readable frames found.")

    frame_indices = np.linspace(
        0, frame_count - 1, SEQUENCE_LENGTH
    ).astype(int)

    pose_detector = None
    hand_detector = None
    features = []

    try:
        pose_detector, hand_detector = create_landmarkers()

        for index, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()

            if not ok:
                features.append(
                    np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
                )
                continue

            height, width = frame.shape[:2]

            if width > MAX_WIDTH:
                new_height = max(
                    1, int(height * MAX_WIDTH / width)
                )
                frame = cv2.resize(
                    frame,
                    (MAX_WIDTH, new_height),
                    interpolation=cv2.INTER_AREA
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = (
                int(frame_index / fps * 1000)
                if fps > 0 else index * 100
            )

            features.append(
                extract_frame_features(
                    rgb,
                    timestamp_ms,
                    pose_detector,
                    hand_detector
                )
            )

    finally:
        cap.release()

        if pose_detector is not None:
            pose_detector.close()

        if hand_detector is not None:
            hand_detector.close()

    sequence = np.stack(features).astype(np.float32)

    if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise ValueError(
            f"Unexpected feature shape: {sequence.shape}"
        )

    return sequence


def predict_sequence(sequence):
    X = np.asarray(sequence, dtype=np.float32).reshape(
        1, SEQUENCE_LENGTH, FEATURES_PER_FRAME
    )

    X = (X - scaler_mean) / scaler_scale
    probabilities = model.predict(X, verbose=0)[0]

    top_indices = np.argsort(probabilities)[-5:][::-1]

    results = {
        str(labels[i]): float(probabilities[i])
        for i in top_indices
    }

    best_index = int(np.argmax(probabilities))
    best_label = str(labels[best_index])
    best_score = float(probabilities[best_index])

    return best_label, best_score, results


def predict_video(video_path):
    try:
        sequence = extract_video_features(video_path)
        best_label, best_score, results = predict_sequence(sequence)

        message = (
            f"Predicted sign: {best_label}\n"
            f"Model score: {best_score:.1%}\n\n"
            "Research prototype: the score is not a guarantee "
            "of prediction correctness."
        )

        return message, results

    except Exception as exc:
        return f"Could not process video: {exc}", {}


def predict_features(text):
    try:
        values = np.fromstring(
            (text or "").replace(",", " "),
            sep=" "
        )

        expected = SEQUENCE_LENGTH * FEATURES_PER_FRAME

        if values.size != expected:
            return {
                "Error": (
                    f"Expected {expected} numbers; "
                    f"received {values.size}."
                )
            }

        if not np.isfinite(values).all():
            return {"Error": "Input contains invalid numeric values."}

        _, _, results = predict_sequence(
            values.reshape(SEQUENCE_LENGTH, FEATURES_PER_FRAME)
        )

        return results

    except Exception as exc:
        return {"Error": str(exc)}


custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
.hero {
    padding: 24px;
    border-radius: 18px;
    background: linear-gradient(120deg, #111827, #243b53);
    color: white;
}
.hero h1 {
    margin-bottom: 8px;
    font-size: 2rem;
}
.hero p {
    opacity: .88;
    margin-bottom: 0;
}
"""

with gr.Blocks(
    title="ASL Signer Independent Recognition",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.HTML("""
    <div class="hero">
        <h1>🤟 ASL Signer Independent Recognition</h1>
        <p>Upload a short sign video for model inference.</p>
    </div>
    """)

    gr.Markdown(
        "The model samples **8 frames**, extracts hand and pose "
        "landmarks, and predicts one of its 100 trained classes."
    )

    with gr.Tab("Recognize a sign from video"):
        video = gr.Video(
            label="Upload ASL video",
            sources=["upload"],
            format="mp4"
        )

        run_button = gr.Button(
            "Analyze sign",
            variant="primary"
        )

        prediction = gr.Textbox(
            label="Main prediction",
            lines=3,
            interactive=False
        )

        top_predictions = gr.Label(
            label="Top 5 model outputs",
            num_top_classes=5
        )

        run_button.click(
            predict_video,
            inputs=video,
            outputs=[prediction, top_predictions]
        )

        
with gr.Tab("Advanced: pre-extracted features"):
    gr.Markdown(
        "Debugging only: enter 2,064 numbers "
        "(8 frames × 258 features)."
    )

    feature_text = gr.Textbox(
        label="Feature values",
        lines=5,
        placeholder="Enter 2064 numeric values..."
    )


            feature_button = gr.Button("Predict from features")

            feature_output = gr.Label(
            label="Top 5 predictions",
            num_top_classes=5
        )

        feature_button.click(
            predict_features,
            inputs=feature_text,
            outputs=feature_output
        )

demo.queue(default_concurrency_limit=1).launch()
