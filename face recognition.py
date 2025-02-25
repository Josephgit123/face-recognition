import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_faces(image_path):
    """Detect faces in an uploaded image"""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if face_locations:
        image = cv2.imread(image_path)
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        result_path = os.path.join(UPLOAD_FOLDER, "detected_" + os.path.basename(image_path))
        cv2.imwrite(result_path, image)
        return result_path
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    detected_image = None
    if request.method == "POST":
        file = request.files["image"]
        if file and file.filename.endswith((".png", ".jpg", ".jpeg")):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            detected_image = detect_faces(filepath)

    return render_template("index.html", detected_image=detected_image)

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
