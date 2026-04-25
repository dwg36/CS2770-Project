def print_keyframe_info(keyframes, keyframe_number=None, window=3):
    """
    If keyframe_number is None:
        print total number of keyframes.

    If keyframe_number is given:
        print the selected keyframe and neighboring keyframes.
        Default: previous 3 + selected + next 3.
    """

    total = len(keyframes)

    if keyframe_number is None:
        print(f"Total keyframes: {total}")
        return total

    if keyframe_number < 0 or keyframe_number >= total:
        raise IndexError(
            f"keyframe_number must be between 0 and {total - 1}, "
            f"but got {keyframe_number}"
        )

    start = max(0, keyframe_number - window)
    end = min(total, keyframe_number + window + 1)

    selected_keyframes = keyframes[start:end]

    for i, kf in enumerate(selected_keyframes, start=start):
        marker = "  <-- selected" if i == keyframe_number else ""

        print(f"Keyframe {i}{marker}")
        print("  frame_idx:", kf["frame_idx"])
        print("  path:", kf["image_path"])
        print("  #keypoints:", len(kf["keypoints"]))
        print("  descriptor shape:", kf["descriptors"].shape)
        print()

    return selected_keyframes

import sys
import pickle
import csv
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_keyframe_info.py <pkl_path> [keyframe_idx]")
        sys.exit(1)

    pkl_path = sys.argv[1]

    with open(pkl_path, "rb") as f:
        keyframes = pickle.load(f)

    if len(sys.argv) == 2:
        print_keyframe_info(keyframes)

    elif len(sys.argv) >= 3 and sys.argv[2] == "list_all":
        output_csv = "keyframe_info_all.csv"

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            #for kf_id in sorted(keyframes.keys()):
            for kf_id, kf in enumerate(keyframes):
                kf = keyframes[kf_id]

                frame_idx = kf["frame_idx"]
                path = kf["image_path"]

                # filename: 1439388638933633.png → timestamp: 1439388638933633
                timestamp = Path(path).stem

                print(f"{kf_id},{frame_idx},{timestamp}")
                writer.writerow([kf_id, frame_idx, timestamp])

        print(f"\nSaved to {output_csv}")
        sys.exit(0)
    else:
        keyframe_idx = int(sys.argv[2])
        print_keyframe_info(keyframes, keyframe_idx)
