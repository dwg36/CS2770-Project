import os
import glob
import pickle
import csv
import cv2
from pathlib import Path

from homography_ransac import (
    extract_orb_features,
    match_descriptors,
    compute_inliers_homography
)


def deserialize_keypoints(kp_list):
    keypoints = []
    for x, y, size, angle, response, octave, class_id in kp_list:
        kp = cv2.KeyPoint(
            float(x),        # x
            float(y),        # y
            float(size),     # size
            float(angle),    # angle
            float(response), # response
            int(octave),     # octave
            int(class_id)    # class_id
        )
        keypoints.append(kp)
    return keypoints


def load_test_to_center_keyframe(csv_path):
    """
    Load CSV mapping from test image timestamp to expected center keyframe index.

    Expected CSV format:
        test_image,center_keyframe
        1440150026135009,412
        1440150026197510,412
    """
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
    """
    Return keyframe indices from center_kf-window to center_kf+window.
    Boundary indices are clipped to [0, total_kf-1].
    """
    start = max(0, center_kf - window)
    end = min(total_kf, center_kf + window + 1)
    return range(start, end)


def load_static_mask(mask_dir, test_timestamp, image_shape):
    """
    Load precomputed dynamic mask and invert it to static mask.

    Expected dynamic mask:
        255 = dynamic object
        0   = static/background

    ORB mask convention:
        255 = allow feature extraction
        0   = block feature extraction
    """
    mask_path = os.path.join(mask_dir, f"{test_timestamp}_mask.png")

    dynamic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if dynamic_mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    if dynamic_mask.shape != image_shape:
        dynamic_mask = cv2.resize(
            dynamic_mask,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    _, dynamic_mask = cv2.threshold(
        dynamic_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    static_mask = cv2.bitwise_not(dynamic_mask)

    return static_mask, mask_path


def compare_test_to_keyframe(
    test_img,
    test_kp,
    test_desc,
    keyframe,
    ratio_thresh=0.75,
    ransac_thresh=3.0
):
    kf_kp = deserialize_keypoints(keyframe["keypoints"])
    kf_desc = keyframe["descriptors"]

    matches = match_descriptors(
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


def evaluate_test_images_against_keyframes(
    test_dir,
    mask_dir=None,
    keyframe_path="baseline_keyframes.pkl",
    mapping_csv="test_image_keyframe_match_converted.csv",
    image_ext="*.png",
    output_csv="test_keyframe_results.csv",
    nfeatures=2000,
    ratio_thresh=0.75,
    ransac_thresh=3.0,
    min_inliers_success=50,
    min_inlier_ratio_success=0.25,
    keyframe_window=5,
    skip_if_mask_missing=False
):
    with open(keyframe_path, "rb") as f:
        keyframes = pickle.load(f)

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

        center_kf = test_to_center_kf[test_timestamp]

        if center_kf < 0 or center_kf >= len(keyframes):
            print(
                f"[SKIP] Invalid center keyframe for {test_timestamp}: "
                f"{center_kf}. Valid range is 0 to {len(keyframes) - 1}"
            )
            continue

        candidate_indices = get_keyframe_window(
            center_kf,
            total_kf=len(keyframes),
            window=keyframe_window
        )
        candidate_indices = list(candidate_indices)

        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        if test_img is None:
            print(f"[WARN] Cannot read test image: {test_path}")
            continue

        static_mask = None
        mask_path = ""
        mask_used = False

        if mask_dir is not None:
            try:
                static_mask, mask_path = load_static_mask(
                    mask_dir=mask_dir,
                    test_timestamp=test_timestamp,
                    image_shape=test_img.shape
                )
                mask_used = True
            except FileNotFoundError as e:
                print(f"[WARN] {e}")

                if skip_if_mask_missing:
                    print(f"[SKIP] Missing mask for: {test_path}")
                    continue

                print(f"[WARN] Use full image without mask: {test_path}")
                static_mask = None
                mask_used = False

        test_kp, test_desc = extract_orb_features(
            test_img,
            mask=static_mask,
            nfeatures=nfeatures
        )

        if test_desc is None or len(test_kp) == 0:
            print(f"[SKIP] No ORB features in test image: {test_path}")
            continue

        best_result = None

        print(
            f"[INFO] test={test_timestamp} | "
            f"center_kf={center_kf} | "
            f"compare_kfs={candidate_indices} | "
            f"mask_used={mask_used}"
        )

        for kf_idx in candidate_indices:
            keyframe = keyframes[kf_idx]

            result = compare_test_to_keyframe(
                test_img=test_img,
                test_kp=test_kp,
                test_desc=test_desc,
                keyframe=keyframe,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh
            )

            # Store actual list index separately because keyframe["frame_idx"] is original frame index.
            result["keyframe_list_idx"] = kf_idx

            if best_result is None:
                best_result = result
            else:
                # Main criterion: more inliers is better.
                # Tie-breaker: higher inlier ratio.
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
            "mask_path": mask_path,
            "mask_used": mask_used,
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
            f"[TEST] {test_path} | "
            f"mask_used={mask_used} | "
            f"center_kf={center_kf} | "
            f"best_kf_list_idx={best_result['keyframe_list_idx']} | "
            f"best_frame_idx={best_result['keyframe_idx']} | "
            f"matches={best_result['num_matches']} | "
            f"inliers={best_result['num_inliers']} | "
            f"inlier_ratio={best_result['inlier_ratio']:.3f} | "
            f"success={success}"
        )

    fieldnames = [
        "test_image",
        "test_timestamp",
        "mask_path",
        "mask_used",
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

    print("===== Test Evaluation Done =====")
    print(f"# test images : {len(test_paths)}")
    print(f"# results     : {len(all_results)}")
    print(f"saved to      : {output_csv}")

    return all_results


if __name__ == "__main__":
    evaluate_test_images_against_keyframes(
        test_dir="../../Datasets/Oxford/Oxford_Images_Validation_2015-08-21-10-40-24/rectified/left",
        mask_dir="../../Datasets/Oxford/sam_dino_mask",
        keyframe_path="baseline_keyframes.pkl",
        mapping_csv="test_image_keyframe_match_converted.csv",
        image_ext="*.png",
        output_csv="test_keyframe_results_sam_dino_masked.csv",
        nfeatures=2000,
        ratio_thresh=0.75,
        ransac_thresh=3.0,
        min_inliers_success=50,
        min_inlier_ratio_success=0.25,
        keyframe_window=5,
        skip_if_mask_missing=False
    )
