import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224)
OUTPUT_DIR = "Tests"
CAPTURED_DIR = "Captured"
MODEL_PATH = "models/jaundice_model.h5"
CLASS_NAMES = ['Normal', 'Jaundiced']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAPTURED_DIR, exist_ok=True)

# Load model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"Model file not found at {MODEL_PATH}")
    exit(1)

def prepare_image_for_model(image_path):
    """Load, resize, normalize, and prepare image for model"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr, img

def predict_image(image_path):
    img_arr, img_pil = prepare_image_for_model(image_path)
    prediction = model.predict(img_arr)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction
    return predicted_class, confidence, img_pil

def save_image_with_prediction(image_path, base_name):
    predicted_class, confidence, img_pil = predict_image(image_path)

    draw = ImageDraw.Draw(img_pil)
    border_color = (255, 0, 0) if predicted_class == 1 else (0, 255, 0)
    border_thickness = 10

    for i in range(border_thickness):
        draw.rectangle(
            [i, i, img_pil.width - i - 1, img_pil.height - i - 1],
            outline=border_color
        )

    font = ImageFont.load_default()
    label = f"{CLASS_NAMES[predicted_class]} ({confidence:.2f})"
    draw.text((10, 10), label, fill=border_color, font=font)

    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{CLASS_NAMES[predicted_class]}.jpg")
    img_pil.save(save_path)
    print(f"Saved labeled image: {save_path}")

def capture_and_process():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam")
        return

    print("Press SPACEBAR to capture the image, or ESC to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Press SPACE to capture", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:  # SPACE pressed
            img_name = os.path.join(CAPTURED_DIR, "captured_image.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Captured image saved as {img_name}")
            save_image_with_prediction(img_name, "captured")
            break

    cam.release()
    cv2.destroyAllWindows()

def upload_and_process():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("No file selected, exiting.")
        return

    print(f"Selected file: {file_path}")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_image_with_prediction(file_path, base_name)

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Capture image from webcam")
    print("2. Upload image file via GUI")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        capture_and_process()
    elif choice == "2":
        upload_and_process()
    else:
        print("Invalid choice. Exiting.")
