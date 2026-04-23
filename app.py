import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import base64
import requests
import io

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e2f;
        color: #e0e0e0;
        font-family: Verdana, sans-serif;
    }
    .canvas-wrapper {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    h1 {
        font-family: Courier New, monospace;
        color: #ffb347;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: 1px solid #ffb347;
        border-radius: 5px;
        padding: 8px 20px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<h1>Handwritten Digit Recognition (0-9)</h1>', unsafe_allow_html=True)
st.write("Draw a digit (0-9) below and click **Predict Digit** to see the model's prediction.")

st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        response = requests.post("http://localhost:5000/predict", json={"image": img_str})
        if response.status_code == 200:
            result = response.json()
            st.subheader(f"Predicted Digit: {result['prediction']}")
            st.bar_chart(result["probabilities"])
        else:
            st.error("Prediction failed: " + response.text)
    else:
        st.warning("Please draw something first.")

st.markdown("---")
st.markdown("Made by Sarvagya Dwivedi")
