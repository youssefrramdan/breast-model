from flask import Flask, request, jsonify
import tflite_runtime.interpreter as tflite
import numpy as np
import requests
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names mapping
CLASS_NAMES = {
    0: "Adenosis",
    1: "Fibroadenoma",
    2: "Lobular Carcinoma",
    3: "Mucinous Carcinoma",
    4: "Papillary Carcinoma",
    5: "Phyllodes Tumor",
    6: "Tubular Adenoma",
    7: "Ductal Carcinoma"
}

def download_and_preprocess_image(image_url):
    # Download image from URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception("Failed to download image")

    # Open image using PIL
    image = Image.open(io.BytesIO(response.content))

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0

    return np.expand_dims(image_array, axis=0)

def predict_image(image_url):
    # Preprocess the image
    try:
        processed_image = download_and_preprocess_image(image_url)
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

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

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image_url is in the request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({'error': 'No image URL provided'}), 400

    image_url = data['image_url']

    try:
        result = predict_image(image_url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to Breast Cancer Classification API',
        'endpoints': {
            'predict': {
                'url': '/predict',
                'method': 'POST',
                'content_type': 'application/json',
                'parameters': {
                    'image_url': 'URL of the image to classify'
                },
                'example_request': {
                    'image_url': 'https://example.com/image.jpg'
                },
                'response': {
                    'class': 'predicted class name',
                    'confidence': 'confidence score (0-1)',
                    'probabilities': 'dictionary of class probabilities'
                }
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
