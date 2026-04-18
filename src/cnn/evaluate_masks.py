# Compares YOLO-generated masks against KITTI ground truth labels and reports IoU scores.
# Usage: python src/evaluate_masks.py

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mask_generator import generate_kitti_masks

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

IMAGE_DIR  = os.path.join(PROJECT_ROOT, 'data', 'semantic', 'training', 'image_2')
LABEL_DIR  = os.path.join(PROJECT_ROOT, 'data', 'semantic', 'training', 'semantic')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'semantic_masks')

DYNAMIC_KITTI_IDS = [24, 25, 26, 27, 28, 29, 30, 32, 33]


def load_ground_truth_mask(label_path):
    """Loads a KITTI semantic label and converts it to a binary dynamic/static mask."""

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if label is None:
        print(f"  WARNING: Could not load label: {label_path}")
        return None

    ground_truth_mask = np.zeros_like(label, dtype=np.uint8)
    for class_id in DYNAMIC_KITTI_IDS:
        ground_truth_mask[label == class_id] = 255

    return ground_truth_mask


def compute_iou(mask_a, mask_b):
    """Computes IoU between two binary masks. Returns a value between 0.0 (no overlap) and 1.0 (perfect match)."""

    a = mask_a == 255
    b = mask_b == 255

    intersection = (a & b).sum()
    union        = (a | b).sum()

    if union == 0:
        return 1.0

    return intersection / union


def run_evaluation(conf=0.25, model_type='yolov8x-seg.pt'):
    """Runs YOLO on all semantic training images, compares against ground truth, and prints IoU scores."""

    if not os.path.isdir(IMAGE_DIR):
        print(f"ERROR: Image folder not found: {IMAGE_DIR}")
        sys.exit(1)

    if not os.path.isdir(LABEL_DIR):
        print(f"ERROR: Label folder not found: {LABEL_DIR}")
        sys.exit(1)

    image_files  = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg')))
    paired_files = []
    for filename in image_files:
        if os.path.exists(os.path.join(LABEL_DIR, filename)):
            paired_files.append(filename)

    if not paired_files:
        print("ERROR: No matching image/label pairs found.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_kitti_masks(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        model_type=model_type,
        conf=conf,
        debug=False
    )

    iou_scores = []
    skipped = 0

    for filename in tqdm(paired_files, desc="Evaluating"):
        yolo_mask = cv2.imread(os.path.join(OUTPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if yolo_mask is None:
            skipped += 1
            continue

        gt_mask = load_ground_truth_mask(os.path.join(LABEL_DIR, filename))
        if gt_mask is None:
            skipped += 1
            continue

        if yolo_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (yolo_mask.shape[1], yolo_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        iou_scores.append(compute_iou(yolo_mask, gt_mask))

    print(f"\n{'='*50}")
    print(f"  Evaluation Results")
    print(f"{'='*50}")
    print(f"  Images evaluated : {len(iou_scores)}")
    print(f"  Images skipped   : {skipped}")
    print(f"  Average IoU      : {np.mean(iou_scores):.3f}  (target: > 0.5 is reasonable, > 0.7 is good)")
    print(f"  Median IoU       : {np.median(iou_scores):.3f}")
    print(f"  Best IoU         : {np.max(iou_scores):.3f}")
    print(f"  Worst IoU        : {np.min(iou_scores):.3f}")
    print(f"{'='*50}")

    sorted_scores = sorted(zip(iou_scores, paired_files))

    print("\n  10 worst performing frames:")
    for score, filename in sorted_scores[:10]:
        print(f"    {filename}  —  IoU: {score:.3f}")

    print("\n  10 best performing frames:")
    for score, filename in sorted_scores[-10:][::-1]:
        print(f"    {filename}  —  IoU: {score:.3f}")


if __name__ == '__main__':
    run_evaluation(conf=0.25, model_type='yolov8x-seg.pt')
