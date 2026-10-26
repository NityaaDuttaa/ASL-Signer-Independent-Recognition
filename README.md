# Signer-Independent ASL Recognition

## 1. Project Overview

This project explores American Sign Language (ASL) sign recognition using
pose and hand landmarks extracted from video sequences. The goal is to
investigate recognition of signs from signers not seen during training.

The current system is a research prototype for individual sign
classification. It does not perform continuous sentence translation.

## 2. Repository Structure

- `notebooks/`: Recovered project notebooks and development history.
- `artifacts/`: Trained model, preprocessing files, experiment summaries,
  diagnostics, and evaluation reports.
- `deployment/`: Gradio application, dependency configuration, and Render
  deployment documentation.

## 3. Research Notebooks

The notebooks represent different development and experimental stages.
They are not guaranteed to run sequentially from a fresh runtime.

- `ASL_Notebook_Part_01.ipynb` through `ASL_Notebook_Part_08.ipynb`:
  Recovered notebook sections.
- `ASL_Notebook_Part_08_REPAIRED.ipynb`: Separately retained repaired
  version of Part 08.
- `Untitled2.ipynb`: Troubleshooting, artifact checks, recovery work,
  model checks, and deployment attempts. Included for project history,
  not as the primary training workflow.

## 4. Model and Input Features

- Sequence length: 8 frames.
- Features per frame: 258.
- Pose features: 132.
- Hand features: 126.
- 100 target classes in the 100-class experiments.
- Model family: Bidirectional LSTM.

## 5. Reported Experimental Results

The available project records report the following for a final LSTM
evaluation:

- Test accuracy: 25.58%.
- Test macro-F1: 0.2443.
- Test weighted-F1: approximately 0.2400.
- Top-5 accuracy: 60.56%.
- Validation accuracy: 32.70%.
- Test set size: 1,075 examples.

An earlier baseline reported 13.12% test accuracy and 0.1088 macro-F1.

These are recorded experimental results, not independently reproduced
results. Confirm the data split and evaluation procedure before comparing
experiments or claiming signer-independent generalization.

## 6. Deployment

Deployment files are in `deployment/`:

- `app.py`: Gradio interface and model inference code.
- `requirements.txt`: Python dependencies.
- `.python-version`: Python version configuration.
- `DEPLOYMENT_STATUS.md`: Deployment setup, known errors, and verification
  checklist.

Demo URL: https://asl-signer-recognition-demo.onrender.com

The latest deployment status has not yet been independently verified.
Consult `deployment/DEPLOYMENT_STATUS.md` before relying on the live demo.

## 7. Limitations and Reproducibility

- Original dataset split provenance and signer separation need verification.
- Some videos or reconstructed features were unavailable.
- Not all notebook cells are guaranteed to run independently.
- The model predicts only among its trained classes.
- A fully reproducible end-to-end run has not been established by collecting
  these files alone.

## 8. Suggested Review Order

1. Part 04: inspect signer-based splitting.
2. Part 05: inspect the 100-class LSTM training.
3. Part 06: inspect evaluation.
4. Part 07: inspect model comparisons and additional experiments.
5. Review `artifacts/` and `deployment/` for model outputs and web-app work.
