from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import joblib
import torch
import torchvision.transforms as transforms
from torchvision import models
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained SVM model
svm_model = joblib.load("injury_classification_svm.pkl")

# Load pre-trained ResNet for feature extraction
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
resnet.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to predict injury status
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(transforms.ToPILImage()(img)).unsqueeze(0)

    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().numpy().flatten()

    prediction = svm_model.predict([feature])[0]
    return "Injured" if prediction == 0 else "Not Injured"

# Flask route for homepage
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get prediction
            result = predict_image(filepath)
            return render_template("index.html", prediction=result, image_path=filepath)

    return render_template("index.html", prediction=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
