from __future__ import annotations

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from card_classifier import load_model, predict_image


st.set_page_config(
    page_title="Playing Card Classifier",
    page_icon="🂠",
    layout="wide",
)


@st.cache_resource
def get_inference_stack():
    device = torch.device("cpu")
    model, class_names, transform = load_model(device=device)
    return model, class_names, transform, device


st.title("Playing Card Classifier")
st.caption("Upload a card image and get the model's top prediction.")

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses your trained EfficientNet-B0 model to classify one of 53 playing card classes."
    )
    st.write("Built with Streamlit for GitHub + Streamlit Community Cloud deployment.")

uploaded_file = st.file_uploader(
    "Choose a playing card image",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to run a prediction.")
    st.stop()

image = Image.open(uploaded_file)
model, class_names, transform, device = get_inference_stack()
predictions = predict_image(
    image=image,
    model=model,
    class_names=class_names,
    transform=transform,
    device=device,
)

top_prediction = predictions[0]

left_col, right_col = st.columns([1, 1])

with left_col:
    st.image(image, caption="Uploaded image", use_container_width=True)

with right_col:
    st.subheader("Prediction")
    st.metric("Best match", str(top_prediction["label"]))
    st.metric("Confidence", f"{float(top_prediction['probability']) * 100:.2f}%")

    chart_data = pd.DataFrame(predictions)
    chart_data["probability_percent"] = chart_data["probability"] * 100
    st.subheader("Top 5 predictions")
    st.bar_chart(chart_data.set_index("label")["probability_percent"])
    st.dataframe(
        chart_data[["label", "probability_percent"]].rename(
            columns={"label": "Card", "probability_percent": "Confidence (%)"}
        ),
        use_container_width=True,
        hide_index=True,
    )

