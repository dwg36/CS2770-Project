import os
import glob
import pickle
import csv
import cv2
import numpy as np
from pathlib import Path

from homography_ransac import compute_inliers_homography


PATCH_SIZE = 14

# keyframe/test feature must use same width
DINO_IMAGE_WIDTH = 672


def deserialize_keypoints(kp_list):
    keypoints = []

    for x, y, size, angle, response, octave, class_id in kp_list:
        keypoints.append(
            cv2.KeyPoint(
                float(x),
                float(y),
                float(size),
                float(angle),
                float(response),
                int(octave),
                int(class_id)
            )
        )

    return keypoints


def load_test_to_center_keyframe(csv_path):
    mapping = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"test_image", "center_keyframe"}
        if reader.fieldnames is None or not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"Mapping CSV must contain columns: {required_cols}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            test_image = str(row["test_image"]).strip()
            center_kf = int(row["center_keyframe"])
            mapping[test_image] = center_kf

    return mapping


def get_keyframe_window(center_kf, total_kf, window=5):
    start = max(0, center_kf - window)
    end = min(total_kf, center_kf + window + 1)
    return range(start, end)


def match_dinov2_descriptors(desc1, desc2, ratio_thresh=0.80):
    if desc1 is None or desc2 is None:
        return []

    if len(desc1) < 2 or len(desc2) < 2:
        return []

    desc1 = np.asarray(desc1, dtype=np.float32)
    desc2 = np.asarray(desc2, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


def compare_test_to_keyframe_dinov2_cached(
    test_kp,
    test_desc,
    keyframe,
    ratio_thresh=0.80,
    ransac_thresh=8.0
):
    kf_kp = deserialize_keypoints(keyframe["keypoints"])
    kf_desc = keyframe["descriptors"]

    matches = match_dinov2_descriptors(
        kf_desc,
        test_desc,
        ratio_thresh=ratio_thresh
    )

    H, inlier_mask, pts1, pts2 = compute_inliers_homography(
        kf_kp,
        test_kp,
        matches,
        ransac_thresh=ransac_thresh
    )

    num_matches = len(matches)

    if inlier_mask is not None:
        num_inliers = int(inlier_mask.sum())
        num_outliers = num_matches - num_inliers
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
    else:
        num_inliers = 0
        num_outliers = num_matches
        inlier_ratio = 0.0

    return {
        "keyframe_idx": keyframe["frame_idx"],
        "keyframe_path": keyframe["image_path"],
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "num_outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "homography": H
    }


def validate_feature_resize_config(keyframes, test_features, expected_width):
    if expected_width is None:
        return

    # If the metadata is missing from the previous `build_dinov2_keyframes.py`, only a warning is printed.
    sample_kf = keyframes[0]
    sample_test = next(iter(test_features.values()))

    kf_width = sample_kf.get("target_width", None)
    test_width = sample_test.get("target_width", None)

    if kf_width is not None and kf_width != expected_width:
        print(
            f"[WARN] keyframe target_width={kf_width}, "
            f"but expected_width={expected_width}"
        )

    if test_width is not None and test_width != expected_width:
        print(
            f"[WARN] test target_width={test_width}, "
            f"but expected_width={expected_width}"
        )

    if kf_width is None:
        print(
            "[WARN] keyframe pkl has no target_width metadata. "
            "Make sure keyframes were built with the same resize width."
        )


def evaluate_test_images_against_keyframes_dinov2_cached(
    test_dir,
    keyframe_path="dinov2_keyframes_w672.pkl",
    test_feature_path="dinov2_test_features_w672.pkl",
    mapping_csv="test_image_keyframe_match_converted.csv",
    image_ext="*.png",
    output_csv="test_keyframe_results_dinov2_cached.csv",
    ratio_thresh=0.80,
    ransac_thresh=8.0,
    min_inliers_success=20,
    min_inlier_ratio_success=0.10,
    keyframe_window=5,
    expected_width=DINO_IMAGE_WIDTH
):
    with open(keyframe_path, "rb") as f:
        keyframes = pickle.load(f)

    with open(test_feature_path, "rb") as f:
        test_features = pickle.load(f)

    validate_feature_resize_config(
        keyframes=keyframes,
        test_features=test_features,
        expected_width=expected_width
    )

    test_to_center_kf = load_test_to_center_keyframe(mapping_csv)
    test_paths = sorted(glob.glob(os.path.join(test_dir, image_ext)))

    if len(test_paths) == 0:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    all_results = []

    for test_path in test_paths:
        test_timestamp = Path(test_path).stem

        if test_timestamp not in test_to_center_kf:
            print(f"[SKIP] No mapping for test image: {test_timestamp}")
            continue

        if test_timestamp not in test_features:
            print(f"[SKIP] No cached DINOv2 feature for: {test_timestamp}")
            continue

        center_kf = test_to_center_kf[test_timestamp]

        if center_kf < 0 or center_kf >= len(keyframes):
            print(
                f"[SKIP] Invalid center keyframe for {test_timestamp}: "
                f"{center_kf}. Valid range is 0 to {len(keyframes) - 1}"
            )
            continue

        candidate_indices = list(
            get_keyframe_window(
                center_kf,
                total_kf=len(keyframes),
                window=keyframe_window
            )
        )

        test_feat = test_features[test_timestamp]
        test_kp = deserialize_keypoints(test_feat["keypoints"])
        test_desc = test_feat["descriptors"]

        if test_desc is None or len(test_kp) == 0:
            print(f"[SKIP] Empty cached DINOv2 feature: {test_timestamp}")
            continue

        best_result = None

        print(
            f"[INFO] test={test_timestamp} | "
            f"center_kf={center_kf} | "
            f"compare_kfs={candidate_indices} | "
            f"features={len(test_kp)} | "
            f"resized={test_feat.get('resized_size')}"
        )

        for kf_idx in candidate_indices:
            keyframe = keyframes[kf_idx]

            result = compare_test_to_keyframe_dinov2_cached(
                test_kp=test_kp,
                test_desc=test_desc,
                keyframe=keyframe,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh
            )

            result["keyframe_list_idx"] = kf_idx

            if best_result is None:
                best_result = result
            else:
                if result["num_inliers"] > best_result["num_inliers"]:
                    best_result = result
                elif (
                    result["num_inliers"] == best_result["num_inliers"]
                    and result["inlier_ratio"] > best_result["inlier_ratio"]
                ):
                    best_result = result

        success = (
            best_result["num_inliers"] >= min_inliers_success
            and best_result["inlier_ratio"] >= min_inlier_ratio_success
        )

        row = {
            "test_image": test_path,
            "test_timestamp": test_timestamp,
            "target_width": test_feat.get("target_width"),
            "resized_size": str(test_feat.get("resized_size")),
            "center_keyframe_idx": center_kf,
            "searched_keyframe_indices": " ".join(map(str, candidate_indices)),
            "best_keyframe_list_idx": best_result["keyframe_list_idx"],
            "best_keyframe_frame_idx": best_result["keyframe_idx"],
            "best_keyframe_path": best_result["keyframe_path"],
            "num_matches": best_result["num_matches"],
            "num_inliers": best_result["num_inliers"],
            "num_outliers": best_result["num_outliers"],
            "inlier_ratio": best_result["inlier_ratio"],
            "success": success
        }

        all_results.append(row)

        print(
            f"[TEST] {test_timestamp} | "
            f"best_kf_list_idx={best_result['keyframe_list_idx']} | "
            f"matches={best_result['num_matches']} | "
            f"inliers={best_result['num_inliers']} | "
            f"inlier_ratio={best_result['inlier_ratio']:.3f} | "
            f"success={success}"
        )

    fieldnames = [
        "test_image",
        "test_timestamp",
        "target_width",
        "resized_size",
        "center_keyframe_idx",
        "searched_keyframe_indices",
        "best_keyframe_list_idx",
        "best_keyframe_frame_idx",
        "best_keyframe_path",
        "num_matches",
        "num_inliers",
        "num_outliers",
        "inlier_ratio",
        "success"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("===== Cached DINOv2 Test Evaluation Done =====")
    print(f"# test images : {len(test_paths)}")
    print(f"# results     : {len(all_results)}")
    print(f"saved to      : {output_csv}")

    return all_results


if __name__ == "__main__":
    evaluate_test_images_against_keyframes_dinov2_cached(
        test_dir="../../Datasets/Oxford/Oxford_Images_Validation_2015-08-21-10-40-24/rectified/left",
        keyframe_path="dinov2_keyframes_w672.pkl",
        test_feature_path="dinov2_test_features_w672.pkl",
        mapping_csv="test_image_keyframe_match_converted.csv",
        image_ext="*.png",
        output_csv="test_keyframe_results_dinov2_cached_w672.csv",
        ratio_thresh=0.80,
        ransac_thresh=8.0,
        min_inliers_success=20,
        min_inlier_ratio_success=0.10,
        keyframe_window=5,
        expected_width=DINO_IMAGE_WIDTH
    )
