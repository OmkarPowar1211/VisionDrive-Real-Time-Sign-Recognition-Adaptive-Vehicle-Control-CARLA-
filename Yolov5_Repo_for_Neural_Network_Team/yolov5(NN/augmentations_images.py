import os
import cv2
import numpy as np
from albumentations import (
    HueSaturationValue, 
    RandomBrightnessContrast,
    CLAHE,
    Compose,
    BboxParams
)
import uuid

# Load YOLO annotations (class, x_center, y_center, width, height)
def load_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    labels = []
    for line in lines:
        cls, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append([x_center, y_center, width, height])
        labels.append(int(cls))
    return bboxes, labels

# Save YOLO annotations
def save_annotations(annotations_path, bboxes, labels):
    with open(annotations_path, 'w') as f:
        for bbox, label in zip(bboxes, labels):
            f.write(f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

# Augmentation pipeline with only non-geometric transforms
def get_augmentation():
    return Compose([
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        # You can add more non-geometry transformations here
    ], bbox_params=BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.0))

# Augment and save
def augment_and_save_image(image_path, annot_path, save_dir):
    if not os.path.exists(annot_path):
        print(f"Skipping {image_path}: annotation not found.")
        return

    image = cv2.imread(image_path)
    bboxes, labels = load_annotations(annot_path)

    augment = get_augmentation()

    try:
        augmented = augment(image=image, bboxes=bboxes, category_ids=labels)
        aug_image = augmented['image']
        # Bounding boxes should be unchanged (only filtered if they go out of bounds)
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['category_ids']

        # Save with unique name
        unique_id = str(uuid.uuid4())[:8]
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        new_img_name = f"{base_filename}_aug_{unique_id}.jpg"
        new_annot_name = f"{base_filename}_aug_{unique_id}.txt"

        cv2.imwrite(os.path.join(save_dir, new_img_name), aug_image)
        save_annotations(os.path.join(save_dir, new_annot_name), aug_bboxes, aug_labels)

        print(f"Saved: {new_img_name}, {new_annot_name}")
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")

# Paths
image_dir = r"C:\Users\athar\Downloads\TH_OWL\AUV\Team7_Carla\Images_Carla_Sim\images\train"
save_dir = r"C:\Users\athar\Downloads\TH_OWL\AUV\Team7_Carla\Images_Carla_Sim\images\augmented"

os.makedirs(save_dir, exist_ok=True)

# Process images
for img_name in os.listdir(image_dir):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(image_dir, img_name)
        annot_path = os.path.join(image_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        augment_and_save_image(img_path, annot_path, save_dir)

print("✅ Augmentation complete with no geometric transforms.")
