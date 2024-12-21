from flask import Flask, render_template, request

# import pytesseract
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("D:/p3/Projects/brain tumor detection.h5")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the input file from the request
    file = request.files["image"]

    # Save the file to a temporary location
    file_path = "../static" + file.filename
    file.save(file_path)

    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Perform inference
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence_percent = np.max(predictions[0]) * 100
    # Interpret the predictions
    class_names = ["No Tumor", "Tumor"]
    predicted_label = class_names[predicted_class]
    print("Predicted Label:", predicted_label)
    print("Confidence Percent:", confidence_percent)
    if predicted_label == "Tumor":
        class1 = "injured"
    else:
        class1 = "not-injured"

    return render_template(
        "result.html",
        result=predicted_label,
        confidence_percent=confidence_percent,
        class1=class1,
    )


if __name__ == "__main__":
    app.run(debug=True, port=4000)
