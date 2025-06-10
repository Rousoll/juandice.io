import os
import shutil
import random
import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)

def split_and_copy(src_dir, train_base_dir, val_base_dir, split_ratio=0.8):
    # Get the class name from the directory name
    class_name = os.path.basename(src_dir)
    
    # List and shuffle images
    images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    random.shuffle(images)
    
    # Calculate split
    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    val_images = images[split_point:]
    
    # Create class subfolders inside train and val
    train_class_dir = os.path.join(train_base_dir, class_name)
    val_class_dir = os.path.join(val_base_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy training images
    for img in train_images:
        shutil.copy(
            os.path.join(src_dir, img),
            os.path.join(train_class_dir, img)
        )

    # Copy validation images
    for img in val_images:
        shutil.copy(
            os.path.join(src_dir, img),
            os.path.join(val_class_dir, img)
        )

    print(f"✅ Copied {len(train_images)} images to {train_class_dir}")
    print(f"✅ Copied {len(val_images)} images to {val_class_dir}")

def extract_eyes(image_path):
    """Detect eyes in an image using Haar cascades and return PIL images of eyes."""
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    eye_images = []
    for (ex, ey, ew, eh) in eyes:
        eye_img = img_cv[ey:ey+eh, ex:ex+ew]
        eye_pil = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
        eye_images.append(eye_pil)

    return eye_images

def prepare_eye_for_model(eye_img, target_size=IMAGE_SIZE):
    """Resize and normalize eye image for model input."""
    eye_img = eye_img.resize(target_size)
    eye_arr = np.array(eye_img).astype('float32') / 255.0
    eye_arr = np.expand_dims(eye_arr, axis=0)
    return eye_arr

# Example usage for splitting dataset
juandice_dir = 'Juandice'  # Check folder name spelling
normal_dir = 'Normal'
train_dir = 'train'
val_dir = 'val'
split_ratio = 0.8  # 80% train, 20% val

split_and_copy(juandice_dir, train_dir, val_dir, split_ratio)
split_and_copy(normal_dir, train_dir, val_dir, split_ratio)
