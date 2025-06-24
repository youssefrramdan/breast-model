from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names mapping
CLASS_NAMES = {
    0: "Adenosis",
    1: "Ductal Carcinoma",
    2: "Fibroadenoma",
    3: "Lobular Carcinoma",
    4: "Mucinous Carcinoma",
    5: "Papillary Carcinoma",
    6: "Phyllodes Tumor",
    7: "Tubular Adenoma"
}

def preprocess_image(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Apply CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)
    lab = cv2.merge((cl, ca, cb))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Normalize
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def predict_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    interpreter.invoke()

    # Get prediction results
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])

    return {
        'class': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(predictions[0])}
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict_image(filepath)
            # Clean up the uploaded file
            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
