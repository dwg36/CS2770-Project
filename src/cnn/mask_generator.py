import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

DYNAMIC_CLASS_IDS = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


def build_mask_fn(conf=0.4, model_type='yolov8s-seg.pt'):
    """Pre-loads the YOLO model once and returns a ready-to-use mask function.
    Pass the result as mask_fn to the pipeline — avoids reloading the model per image."""
    model = YOLO(model_type)
    def _fn(image_path):
        return _build_mask(model, image_path, conf)
    return _fn


def generate_mask(image_path, conf=0.4, model_type='yolov8s-seg.pt'):
    model = YOLO(model_type)
    return _build_mask(model, image_path, conf)


def generate_masks(image_dir, output_dir, model_type='yolov8s-seg.pt',
                   conf=0.4, debug=False, debug_dir=None):
    """Runs the model on all images in image_dir and saves binary masks to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    model = YOLO(model_type)
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg')))

    for img_name in tqdm(image_files, desc="Generating Masks"):
        img_path = os.path.join(image_dir, img_name)
        final_mask = _build_mask(model, img_path, conf)
        cv2.imwrite(os.path.join(output_dir, img_name), final_mask)

        if debug and debug_dir:
            original_image = cv2.imread(img_path)
            overlay = original_image.copy()
            overlay[final_mask == 255] = [0, 0, 255]
            blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
            cv2.imwrite(os.path.join(debug_dir, img_name), blended)


def _build_mask(model, image_path, conf):
    """Shared by all mask functions — runs model on one image and returns the mask array."""
    results = model(image_path, verbose=False, conf=conf)
    result = results[0]

    img_height, img_width = result.orig_shape
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    if result.masks is not None:
        for mask_data, cls in zip(result.masks.data, result.boxes.cls):
            if int(cls.item()) in DYNAMIC_CLASS_IDS:
                mask_np = mask_data.cpu().numpy()
                mask_np = cv2.resize(mask_np, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                mask = np.logical_or(mask, mask_np > 0.5).astype(np.uint8)

    return mask * 255
