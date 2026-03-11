import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from training.model import LungCancerCNN
from utils.preprocessing import preprocess_image

app = Flask(__name__)
CORS(app)

# ResNet18 6-class model (best checkpoint)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'lung_model_6class_best.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LungCancerCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
model.to(device)

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Malignant"
]

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    temp_path = "temp_image.png"
    img_file.save(temp_path)

    try:
        image_tensor = preprocess_image(temp_path).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        prediction = CLASS_NAMES[predicted_class]
        confidence = probabilities[0][predicted_class].item()

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)
