# Render Deployment Status

## Demo URL

https://asl-signer-recognition-demo.onrender.com

## Deployment Files

- `app.py`: Gradio web interface and ASL model inference.
- `requirements.txt`: Python dependencies for the application.
- `.python-version`: Specifies Python 3.11.11.

## Intended Application Workflow

1. The user uploads a video.
2. The application samples 8 frames.
3. MediaPipe extracts pose and hand landmarks.
4. The application constructs 258 features per frame.
5. The trained LSTM predicts a sign class and returns the top predictions.

The application is intended to classify individual signs, not translate
continuous sign-language sentences.

## Deployment and Debugging History

- The application was configured for deployment on Render.
- An earlier deployment failed because Markdown code fences were
  accidentally included in `app.py`. These were removed.
- A later deployment encountered a `NameError` involving `feature_text`.
- A code correction was attempted, but the final deployment status has
  not yet been independently verified.

## Current Status

The deployment files have been collected in this repository handoff.
The application should be tested before claiming that the live demo
works correctly.

## Next Verification Steps

1. Confirm that Render finishes building and starting the application.
2. Open the live URL and check that the Gradio interface loads.
3. Upload a short, supported ASL video.
4. Verify that prediction completes and returns meaningful output.
5. Check the Render logs if the application fails.

## Limitations

- Prediction quality depends on the trained model and its data.
- The model recognizes only its trained classes.
- Signer-independent generalization must be validated using a properly
  verified signer-disjoint test set.
