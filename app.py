from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('emotion_model.h5')

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make prediction."""
    try:
        # Check if file part exists
        if 'file' not in request.files:
            return "No file part in request", 400

        file = request.files['file']

        if file.filename == '':
            return "No file selected", 400

        if not allowed_file(file.filename):
            return "Invalid file format. Please upload a PNG or JPG image.", 400

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Open and preprocess the image safely
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))
        img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        emotion_index = np.argmax(prediction)

        # Define emotion labels
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predicted_emotion = emotions[emotion_index]

        # Render result page
        return render_template('result.html', emotion=predicted_emotion, image_path=filepath)

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
