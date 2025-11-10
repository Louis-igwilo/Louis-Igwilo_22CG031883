import os
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import uuid

# Initialize Flask app
app = Flask(__name__, static_folder="uploads")  # serve uploads as static

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
# Database table
class UserEmotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    image_filename = db.Column(db.String(100))
    prediction = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Create table if not exists
with app.app_context():
    db.create_all()

# Load your trained model
MODEL_PATH = "emotion_model.h5"
model = load_model(MODEL_PATH)

# Define emotion classes
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Folder to save uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to save record
def save_record(name, image_filename, prediction):
    record = UserEmotion(name=name, image_filename=image_filename, prediction=prediction)
    db.session.add(record)
    db.session.commit()

# -----------------------
# Routes
# -----------------------

# Homepage: shows form
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Predict route: handles POST and shows result
@app.route("/predict", methods=["POST"])
def predict():
    name = request.form.get("name", "Anonymous")
    file = request.files.get("file")  # must match input name in HTML
    if file:
        # Save uploaded image
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Prepare image for model
        img = Image.open(filepath).convert('L')  # grayscale
        img = img.resize((48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict emotion
        prediction_probs = model.predict(img_array)
        prediction = EMOTIONS[np.argmax(prediction_probs)]

        # Save record in database
        save_record(name, filename, prediction)

        # Render result page
        return render_template("result.html",
                               emotion=prediction,
                               image_path=filename)

    # Redirect to homepage if no file
    return redirect("/")

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
