# Run this to generate YOLO masks for a KITTI sequence. Usage: python src/run_sequence.py --sequence 00 --debug

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from mask_generator import generate_kitti_masks

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'sequences')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'output', 'masks')
DEBUG_DIR    = os.path.join(PROJECT_ROOT, 'output', 'debug')


def run_sequence(sequence_id, debug=False, conf=0.4, model='yolov8s-seg.pt'):
    """Sets up paths for a sequence and calls the mask generator."""

    image_dir = os.path.join(DATA_DIR, sequence_id, 'image_2')

    if not os.path.isdir(image_dir):
        print(f"ERROR: Could not find images at: {image_dir}")
        print(f"       Make sure sequence {sequence_id} is extracted into data/sequences/{sequence_id}/image_2/")
        return False

    output_dir = os.path.join(OUTPUT_DIR, sequence_id)
    debug_dir  = os.path.join(DEBUG_DIR, sequence_id) if debug else None

    start_time = time.time()

    generate_kitti_masks(
        image_dir=image_dir,
        output_dir=output_dir,
        model_type=model,
        conf=conf,
        debug=debug,
        debug_dir=debug_dir,
    )

    elapsed_seconds = time.time() - start_time
    num_masks = len([f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg'))])
    print(f"Sequence {sequence_id}: {num_masks} masks in {elapsed_seconds:.1f}s ({elapsed_seconds / num_masks:.2f}s/frame)")
    return True


def find_available_sequences():
    """Returns a sorted list of sequence IDs found in data/sequences/."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate YOLO masks for KITTI sequences')

    parser.add_argument('--sequence', nargs='+', default=None,
                        help='Which sequence(s) to process. Example: --sequence 00  or  --sequence 00 01 05. '
                             'Defaults to all sequences found in data/sequences/')
    parser.add_argument('--debug', action='store_true',
                        help='Also save red overlay images so you can visually check the masks')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='YOLO confidence threshold between 0 and 1 (default: 0.4)')
    parser.add_argument('--model', type=str, default='yolov8s-seg.pt',
                        help='Which YOLO model to use (default: yolov8s-seg.pt)')

    args = parser.parse_args()

    sequences = args.sequence
    if sequences is None:
        sequences = find_available_sequences()
        if not sequences:
            print(f"ERROR: No sequences found in {DATA_DIR}")
            sys.exit(1)
        print(f"Processing sequences: {', '.join(sequences)}")

    for seq in sequences:
        seq = seq.zfill(2)
        success = run_sequence(seq, debug=args.debug, conf=args.conf, model=args.model)
        if not success:
            sys.exit(1)
