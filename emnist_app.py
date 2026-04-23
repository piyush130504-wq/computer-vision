import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

# EMNIST Balanced label map (47 classes)
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# App title and instructions
st.title("Handwritten Character Recognition (EMNIST Balanced)")
st.write("Draw a digit or letter below and click **Predict Character** to see the model's prediction.")

# Drawing canvas
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

# Prediction logic
if st.button("Predict Character"):
    if canvas_result.image_data is not None:
        # Convert to grayscale PIL image
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'), mode='L')

        # Fix EMNIST rotation/flip issue
        img = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)

        # Resize and normalize
        img = img.resize((28, 28))
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 28, 28, 1)

        # Load EMNIST model
        model = tf.keras.models.load_model("emnist_balanced_cnn.h5")
        prediction = model.predict(img_arr)
        pred_class = int(np.argmax(prediction))
        predicted_char = emnist_labels[pred_class]

        # Display prediction
        st.subheader(f"Predicted Character: `{predicted_char}`")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw something first.")

# Footer
st.markdown("---")
st.caption("Made by Sarvagya Dwivedi")
