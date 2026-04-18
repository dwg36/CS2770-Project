# Removes ORB keypoints that land on dynamic objects using the YOLO mask.

import cv2
import numpy as np


# Returns only keypoints on static (black) regions of the mask.
def filter_keypoints(keypoints, mask):
    if mask is None or len(keypoints) == 0:
        return keypoints

    static_keypoints = []

    for kp in keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        x = max(0, min(x, mask.shape[1] - 1))
        y = max(0, min(y, mask.shape[0] - 1))

        if mask[y, x] == 0:
            static_keypoints.append(kp)

    return static_keypoints


# Same as filter_keypoints but keeps descriptors aligned with their keypoints.
def filter_keypoints_and_descriptors(keypoints, descriptors, mask):
    if mask is None or len(keypoints) == 0:
        return keypoints, descriptors

    indices_to_keep = []

    for i in range(len(keypoints)):
        x = max(0, min(int(round(keypoints[i].pt[0])), mask.shape[1] - 1))
        y = max(0, min(int(round(keypoints[i].pt[1])), mask.shape[0] - 1))

        if mask[y, x] == 0:
            indices_to_keep.append(i)

    if len(indices_to_keep) == 0:
        empty_descriptors = np.empty((0, descriptors.shape[1]), dtype=descriptors.dtype)
        return [], empty_descriptors

    filtered_keypoints = [keypoints[i] for i in indices_to_keep]
    filtered_descriptors = descriptors[indices_to_keep]

    return filtered_keypoints, filtered_descriptors


# Returns a summary string of how many keypoints were removed.
def filtering_stats(original_keypoints, filtered_keypoints):
    total = len(original_keypoints)
    kept = len(filtered_keypoints)
    removed = total - kept

    if total == 0:
        percentage = 0
    else:
        percentage = (removed / total) * 100

    return f"Keypoints: {total} total → {kept} kept ({removed} removed, {percentage:.1f}% were on dynamic objects)"
