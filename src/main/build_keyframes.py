import os
import glob
import pickle
import cv2

from homography_ransac import (
    extract_orb_features,
    match_descriptors,
    compute_inliers_homography
)

def serialize_keypoints(keypoints):
    return [
        (
            kp.pt[0], kp.pt[1],  # x, y
            kp.size,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id
        )
        for kp in keypoints
    ]

def make_keyframe_record(image_path, frame_idx, image_shape, keypoints, descriptors):
    return {
        "frame_idx": frame_idx,
        "image_path": image_path,
        "image_shape": image_shape,
        "keypoints": serialize_keypoints(keypoints),
        "descriptors": descriptors
    }


def build_baseline_keyframes(
    baseline_dir,
    image_ext="*.png",
    save_path="baseline_keyframes.pkl",
    nfeatures=2000,
    ratio_thresh=0.75,
    ransac_thresh=3.0,
    min_inliers_to_skip=120,
    min_frame_gap=5
):
    image_paths = sorted(glob.glob(os.path.join(baseline_dir, image_ext)))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {baseline_dir}")

    keyframes = []

    last_kf_idx = None
    last_kf_kp = None
    last_kf_desc = None

    for frame_idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue

        kp, desc = extract_orb_features(img, nfeatures=nfeatures)

        if desc is None or len(kp) == 0:
            print(f"[SKIP] No ORB features: {image_path}")
            continue

        # First valid frame is always selected as a keyframe
        if len(keyframes) == 0:
            keyframes.append(
                make_keyframe_record(
                    image_path=image_path,
                    frame_idx=frame_idx,
                    image_shape=img.shape,
                    keypoints=kp,
                    descriptors=desc
                )
            )

            last_kf_idx = frame_idx
            last_kf_kp = kp
            last_kf_desc = desc

            print(f"[KEYFRAME] frame={frame_idx}, first keyframe")
            continue

        # Avoid selecting keyframes too frequently
        if frame_idx - last_kf_idx < min_frame_gap:
            print(f"[SKIP] frame={frame_idx}, too close to last keyframe")
            continue

        # Compare current baseline frame with last selected keyframe
        matches = match_descriptors(
            last_kf_desc,
            desc,
            ratio_thresh=ratio_thresh
        )

        H, inlier_mask, pts1, pts2 = compute_inliers_homography(
            last_kf_kp,
            kp,
            matches,
            ransac_thresh=ransac_thresh
        )

        if inlier_mask is not None:
            num_inliers = int(inlier_mask.sum())
            num_matches = len(matches)
            inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
        else:
            num_inliers = 0
            num_matches = len(matches)
            inlier_ratio = 0.0

        # Low inlier count means current frame is sufficiently different
        # from the last keyframe, so save it as a new keyframe.
        if num_inliers < min_inliers_to_skip:
            keyframes.append(
                make_keyframe_record(
                    image_path=image_path,
                    frame_idx=frame_idx,
                    image_shape=img.shape,
                    keypoints=kp,
                    descriptors=desc
                )
            )

            last_kf_idx = frame_idx
            last_kf_kp = kp
            last_kf_desc = desc

            print(
                f"[KEYFRAME] frame={frame_idx}, "
                f"matches={num_matches}, inliers={num_inliers}, "
                f"inlier_ratio={inlier_ratio:.3f}"
            )
        else:
            print(
                f"[SKIP] frame={frame_idx}, "
                f"matches={num_matches}, inliers={num_inliers}, "
                f"inlier_ratio={inlier_ratio:.3f}"
            )

    with open(save_path, "wb") as f:
        pickle.dump(keyframes, f)

    print("===== Baseline Keyframe Build Done =====")
    print(f"# baseline images : {len(image_paths)}")
    print(f"# keyframes       : {len(keyframes)}")
    print(f"saved to          : {save_path}")

    return keyframes


if __name__ == "__main__":
    build_baseline_keyframes(
        baseline_dir="../../Datasets/Oxford/Oxford_Images_Base_2015-08-12-15-04-18/rectified/left",
        image_ext="*.png",
        save_path="baseline_keyframes.pkl",
        nfeatures=2000,
        ratio_thresh=0.75,
        ransac_thresh=3.0,
        min_inliers_to_skip=120,
        min_frame_gap=5
    )
