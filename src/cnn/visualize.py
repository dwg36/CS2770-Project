# Shows a 3-panel demo: all keypoints | YOLO mask | filtered keypoints. Usage: python src/cnn/visualize.py --sequence 00

import argparse
import os
import sys
import random
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
from mask_generator import generate_mask
from feature_filter import filter_keypoints, filtering_stats
from homography_ransac import extract_orb_features

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'output')


# Draws keypoints as green squares with a title bar and optional bottom label.
def draw_keypoints_panel(image_bgr, keypoints, title, bottom_label=None):
    panel = image_bgr.copy()

    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        cv2.rectangle(panel, (x - 4, y - 4), (x + 4, y + 4), (0, 255, 0), 1)

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if bottom_label is not None:
        cv2.rectangle(panel, (0, panel.shape[0] - 28), (panel.shape[1], panel.shape[0]), (20, 20, 20), -1)
        cv2.putText(panel, bottom_label, (8, panel.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return panel


# Draws the mask as a red overlay with contour outlines around dynamic regions.
def draw_mask_panel(image_bgr, mask):
    original = image_bgr.copy()
    red_overlay = image_bgr.copy()
    red_overlay[mask == 255] = [0, 0, 220]  # BGR — this is red
    panel = cv2.addWeighted(original, 0.6, red_overlay, 0.4, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(panel, [contour], -1, (0, 0, 180), 2)

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(panel, 'YOLO Mask  (red = dynamic, will be filtered)',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return panel


# Builds the 3-panel comparison image for one frame.
def build_panel(image_path, mask=None, conf=0.4):
    image_bgr  = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if mask is None:
        mask = generate_mask(image_path, conf=conf)

    all_keypoints, _ = extract_orb_features(image_gray)
    static_keypoints = filter_keypoints(all_keypoints, mask)
    print(f"  {filtering_stats(all_keypoints, static_keypoints)}")

    n_all     = len(all_keypoints)
    n_static  = len(static_keypoints)
    n_removed = n_all - n_static

    panel_before = draw_keypoints_panel(
        image_bgr, all_keypoints,
        title=f'Before filter  -  {n_all} keypoints',
        bottom_label='Without semantic filtering'
    )
    panel_mask = draw_mask_panel(image_bgr, mask)
    panel_after = draw_keypoints_panel(
        image_bgr, static_keypoints,
        title=f'After filter  -  {n_static} keypoints',
        bottom_label=f'{n_removed} dynamic keypoints removed  ({n_removed / n_all * 100:.1f}%)'
    )

    divider = np.full((image_bgr.shape[0], 3, 3), 60, dtype=np.uint8)
    return np.hstack([panel_before, divider, panel_mask, divider, panel_after])


def run_single_image(image_path, mask_path=None, save_path=None, conf=0.4):
    mask = None
    if mask_path is not None and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    panel = build_panel(image_path, mask=mask, conf=conf)

    if save_path is not None:
        save_folder = os.path.dirname(save_path)
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
        cv2.imwrite(save_path, panel)
    else:
        cv2.imshow('Semantic Filter - Before | Mask | After', panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Picks n random frames from a sequence and shows the 3-panel visualizer for each.
def run_sequence_sample(sequence_id, subdir='', n_samples=5, save_dir=None, conf=0.4):
    image_dir = os.path.join(PROJECT_ROOT, 'data', 'sequences', sequence_id, subdir) if subdir \
                else os.path.join(PROJECT_ROOT, 'data', 'sequences', sequence_id)
    mask_dir  = os.path.join(PROJECT_ROOT, 'output', 'masks', sequence_id)

    if not os.path.isdir(image_dir):
        print(f"ERROR: Image folder not found: {image_dir}")
        sys.exit(1)

    all_images    = sorted(f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg')))
    sample_images = random.sample(all_images, min(n_samples, len(all_images)))

    for filename in sample_images:
        img_path  = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        panel = build_panel(img_path, mask=mask, conf=conf)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f'panel_{filename}'), panel)
        else:
            cv2.imshow(f'Semantic Filter - {filename}', panel)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:  # ESC
                break


# Returns the first available sequence ID from data/sequences/.
def find_first_sequence():
    seq_dir = os.path.join(PROJECT_ROOT, 'data', 'sequences')
    if not os.path.isdir(seq_dir):
        return None
    sequences = sorted(d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d)))
    return sequences[0] if sequences else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize semantic feature filtering')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--image',    type=str, help='Path to a single image file')
    group.add_argument('--sequence', type=str, help='Sequence ID, e.g. 00. Defaults to the first available sequence.')

    parser.add_argument('--subdir', type=str,   default='',   help='Subdirectory inside sequence folder containing images (e.g. image_2 for KITTI)')
    parser.add_argument('--mask',   type=str,   default=None, help='Path to a pre-generated mask (optional)')
    parser.add_argument('--save',   type=str,   default=None, help='Save the panel to this file path instead of showing it')
    parser.add_argument('--sample', type=int,   default=5,    help='How many random frames to show in sequence mode (default: 5)')
    parser.add_argument('--conf',   type=float, default=0.4,  help='YOLO confidence threshold (default: 0.4)')

    args = parser.parse_args()

    if args.image:
        if args.save is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fname = os.path.splitext(os.path.basename(args.image))[0]
            args.save = os.path.join(OUTPUT_DIR, f'panel_{fname}.jpg')
            print(f"Saving to: {args.save}")
        run_single_image(args.image, mask_path=args.mask, save_path=args.save, conf=args.conf)
    else:
        sequence = args.sequence
        if sequence is None:
            sequence = find_first_sequence()
            if sequence is None:
                print(f"ERROR: No sequences found in data/sequences/")
                print(f"       Run run_sequence.py first, or pass --sequence 00 or --image <path>")
                sys.exit(1)
            print(f"Using sequence: {sequence}")
        run_sequence_sample(sequence, subdir=args.subdir, n_samples=args.sample, save_dir=args.save, conf=args.conf)
