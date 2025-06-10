```markdown
# Jaundice Eye Detection - Dataset Preparation & Preprocessing

This project is focused on preparing a medical image dataset for detecting jaundice from eye images. It includes splitting datasets into training and validation sets, extracting eyes from facial images using Haar cascades, and formatting the data for machine learning models.

---

## 📁 Directory Structure

Before running the script, structure your dataset like this:

```

dataset/
├── Jaundice/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Normal/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...

```

After processing, you'll get:

```

train/
├── Jaundice/
├── Normal/

val/
├── Jaundice/
├── Normal/

````

---

## 📦 Requirements

Make sure you have the following Python packages installed:

```bash
pip install opencv-python 
pip install numpy 
pip install pillow 
pip install tensorflow
````

---

## 🚀 How to Run

1. Place your dataset inside folders named `Jaundice` and `Normal`.
2. Run the class of data_processing.py
OR
3. Run the preprocessing script:

```bash
python data_preprocessing_haar.py
```

This will:

* Split the data into training and validation sets (default 80/20 split)
* Copy images to new `train/` and `val/` directories
* Provide utility functions to extract and prepare eye regions for model input

---

## 📌 Functions Overview

### `split_and_copy(src_dir, train_base_dir, val_base_dir, split_ratio)`

Splits images into train and validation folders while keeping class labels.

### `extract_eyes(image_path)`

Uses Haar cascades to detect eyes and return them as cropped PIL images.

### `prepare_eye_for_model(eye_img)`

Resizes and normalizes eye images to prepare for model input.

---

## 🧠 Next Step

After preprocessing, the images are ready for training a CNN or any other classification model to detect jaundice based on eye images.

---

## 📝 Notes

* The current method uses OpenCV's Haar cascade for eye detection, which may not work well in poor lighting or occlusions.
* Make sure your dataset has clear eye visibility to improve model performance.

---

## 👨‍💻 Author

Developed by \ Rousol Sabobeh
