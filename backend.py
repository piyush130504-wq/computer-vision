from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = load_model("mnist_strong_cnn1.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    image_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({
        "prediction": predicted_digit,
        "probabilities": prediction[0].tolist()
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
