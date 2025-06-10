from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Run your existing eye detection + prediction pipeline here
    # Example:
    # prediction = run_model_on_image(img)
    # For demo, let's say:
    prediction = "Jaundice"
    confidence = 0.85

    return jsonify({'prediction': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
