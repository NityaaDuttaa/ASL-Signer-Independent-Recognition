# ASL Signer-Independent Recognition

This folder contains the research handoff for our ASL sign recognition project.

## Contents

- `notebooks/`: Recovered research notebooks and development history.
- `artifacts/`: Trained model, preprocessing files, and evaluation reports.
- `deployment/`: Gradio application, dependencies, and Render deployment documentation.

## Model Overview

The prototype uses pose and hand landmarks extracted from video sequences. The model processes 8 frames with 258 features per frame and predicts among 100 sign classes.

## Reported Results

- Test accuracy: 25.58%
- Macro-F1: 0.2443
- Top-5 accuracy: 60.56%

These are recorded experimental results and have not been independently reproduced. Signer-independent generalization still requires verification.

## Deployment

Demo URL: https://asl-signer-recognition-demo.onrender.com

See `deployment/DEPLOYMENT_STATUS.md` for deployment details and known issues. The live application has not yet been confirmed to work correctly.

## Important Note

The notebooks represent different experimental stages and may not run sequentially from a fresh environment. This handoff preserves the existing work for review and continuation.
