import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
import mediapipe as mp

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "models/jaundice_model.h5"
CLASS_NAMES = ['Normal', 'Jaundiced']
CAPTURED_DIR = "Captured"
image_paths = glob.glob(os.path.join(CAPTURED_DIR, "*"))

# Mediapipe Face Mesh initialization
mp_face_mesh = mp.solutions.face_mesh

def extract_eyes_with_coords(image_path, padding=5):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    results = face_mesh.process(img_rgb)
    eyes_data = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246]
            left_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466]

            def crop_eye(indices):
                xs = [int(face_landmarks.landmark[i].x * w) for i in indices]
                ys = [int(face_landmarks.landmark[i].y * h) for i in indices]

                x_min = max(min(xs) - padding, 0)
                y_min = max(min(ys) - padding, 0)
                x_max = min(max(xs) + padding, w)
                y_max = min(max(ys) + padding, h)

                eye_img = img[y_min:y_max, x_min:x_max]
                eye_pil = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
                return eye_pil, (x_min, y_min, x_max - x_min, y_max - y_min)

            left_eye = crop_eye(left_eye_indices)
            right_eye = crop_eye(right_eye_indices)

            eyes_data.extend([left_eye, right_eye])

    face_mesh.close()
    return eyes_data


def prepare_eye_for_model(eye_img, target_size=IMAGE_SIZE):
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    eye_img = eye_img.resize(target_size, resample)
    eye_arr = np.array(eye_img).astype('float32') / 255.0
    eye_arr = np.expand_dims(eye_arr, axis=0)
    return eye_arr


def create_data_generator(directory):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow_from_directory(
        directory,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )


def create_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_gen, val_gen):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    return history


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        model = models.load_model(path)
        print(f"Model loaded from {path}")
        return model
    else:
        print("Model not found.")
        return None


# New function to analyze sclera color for yellow/red tints
def analyze_sclera_color(eye_img_pil):
    """
    Analyze the color of the sclera (white part) in the eye image.
    Returns:
        color_label: 'yellow', 'red', or 'normal'
    """
    eye_img = cv2.cvtColor(np.array(eye_img_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(eye_img, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([15, 40, 40])
    yellow_upper = np.array([35, 255, 255])

    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])

    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    yellow_ratio = np.sum(mask_yellow > 0) / (eye_img.shape[0] * eye_img.shape[1])
    red_ratio = np.sum(mask_red > 0) / (eye_img.shape[0] * eye_img.shape[1])

    # Thresholds to tune as needed
    if yellow_ratio > 0.05:
        return 'yellow'
    elif red_ratio > 0.05:
        return 'red'
    else:
        return 'normal'


# Modified test function integrating color analysis
def test_model_on_eyes_with_boxes(model, image_path):
    eye_data = extract_eyes_with_coords(image_path)
    if not eye_data:
        print(f"No eyes detected in {image_path}")
        return None

    predictions = []
    color_labels = []

    for eye_img, coords in eye_data:
        eye_arr = prepare_eye_for_model(eye_img)
        pred = model.predict(eye_arr)[0][0]
        predictions.append(pred)

        color_label = analyze_sclera_color(eye_img)
        color_labels.append(color_label)

    # Logic to combine color analysis with model prediction
    if 'yellow' in color_labels:
        predicted_class = 1
        confidence = 1.0
        reason = "Yellow sclera detected → Jaundice"
    elif 'red' in color_labels:
        predicted_class = 0
        confidence = 1.0
        reason = "Red sclera detected → Possible infection or dehydration"
    else:
        avg_pred = np.mean(predictions)
        predicted_class = 1 if avg_pred > 0.5 else 0
        confidence = avg_pred if predicted_class == 1 else 1 - avg_pred
        reason = "Model prediction"

    # Draw results on image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    label_text = f"{CLASS_NAMES[predicted_class]} ({confidence:.2f})"
    draw.text((10, 10), label_text, fill=(255, 0, 0), font=font)

    for (eye_img, (ex, ey, ew, eh)), pred, color_label in zip(eye_data, predictions, color_labels):
        eye_class = 1 if pred > 0.5 else 0
        box_color = (255, 0, 0) if eye_class == 1 else (0, 255, 0)
        if color_label == 'yellow':
            box_color = (255, 255, 0)  # Yellow box
        elif color_label == 'red':
            box_color = (255, 0, 0)    # Red box (same as jaundice, but means infection)

        thickness = 4
        for i in range(thickness):
            draw.rectangle(
                [ex - i, ey - i, ex + ew + i, ey + eh + i],
                outline=box_color
            )

    os.makedirs("Tests", exist_ok=True)
    output_path = os.path.join("Tests", f"{CLASS_NAMES[predicted_class]}_{os.path.basename(image_path)}")
    image.save(output_path)

    print(f"Predicted class: {label_text} ({reason})")
    print(f"Saved labeled image with eye boxes to {output_path}")
    return predicted_class


def main():
    train_dir = 'train'
    val_dir = 'val'

    print("Preparing data...")
    train_gen = create_data_generator(train_dir)
    val_gen = create_data_generator(val_dir)

    print("Building model...")
    model = create_model()
    compile_model(model)

    print("Training model...")
    history = train_model(model, train_gen, val_gen)

    print("Saving model...")
    save_model(model)

    return model, history


if __name__ == "__main__":
    model, history = main()

    if not image_paths:
        print(f"No images found in {CAPTURED_DIR}")
    else:
        for image_path in image_paths:
            print(f"Testing {image_path}")
            test_model_on_eyes_with_boxes(model, image_path)
