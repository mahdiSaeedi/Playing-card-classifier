# Playing Card Classifier

This project contains a trained PyTorch playing card classifier and a Streamlit web app for running predictions from uploaded images.

## What's in the repo

- `app.py`: Streamlit web app
- `card_classifier.py`: model architecture and inference helpers
- `class_names.json`: the 53 output labels
- `Code and Data/Output/card_classifier.pth`: trained model weights
- `Code and Data/playing-cards-model-training.ipynb`: training notebook

The raw training dataset is excluded from GitHub so the repo stays small and deployable.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app from this repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Notes

- The app expects the trained model file at `Code and Data/Output/card_classifier.pth`.
- The current model predicts 53 classes, including all suits and ranks plus `joker`.
