import sys
import os
import pickle
import cv2
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
from homography_ransac import compute_inliers_homography, draw_inlier_outlier_matches


PATCH_SIZE = 14
DINO_IMAGE_WIDTH = 672
KEYFRAME_PKL = "dinov2_keyframes_w672.pkl"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dinov2_model(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(device)
    return model


def resize_keep_aspect(image_bgr, target_width):
    if target_width is None:
        return image_bgr, 1.0

    h, w = image_bgr.shape[:2]
    scale = target_width / float(w)
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image_bgr,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def preprocess_image_for_dino(image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    h, w = image_rgb.shape[:2]

    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE

    image_rgb = np.pad(
        image_rgb,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0
    )

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_rgb = (image_rgb - mean) / std

    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)

    return tensor, h, w, image_rgb.shape[0], image_rgb.shape[1]


def extract_dinov2_features_as_keypoints(image_bgr, model, device):
    tensor, orig_h, orig_w, padded_h, padded_w = preprocess_image_for_dino(
        image_bgr,
        device
    )

    with torch.no_grad():
        features = model.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"][0]

    grid_h = padded_h // PATCH_SIZE
    grid_w = padded_w // PATCH_SIZE

    patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=1)
    patch_tokens = patch_tokens.cpu().numpy().astype(np.float32)

    keypoints = []
    descriptors = []

    idx = 0

    for gy in range(grid_h):
        for gx in range(grid_w):
            cx = gx * PATCH_SIZE + PATCH_SIZE / 2.0
            cy = gy * PATCH_SIZE + PATCH_SIZE / 2.0

            if cx < orig_w and cy < orig_h:
                keypoints.append(
                    cv2.KeyPoint(
                        float(cx),
                        float(cy),
                        float(PATCH_SIZE)
                    )
                )
                descriptors.append(patch_tokens[idx])

            idx += 1

    descriptors = np.asarray(descriptors, dtype=np.float32)

    return keypoints, descriptors


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


def compare_test_image_to_dinov2_keyframe(
    test_image_path,
    keyframe_number,
    keyframe_pkl=KEYFRAME_PKL,
    target_width=DINO_IMAGE_WIDTH,
    ratio_thresh=0.80,
    ransac_thresh=8.0,
    max_draw=100
):
    with open(keyframe_pkl, "rb") as f:
        keyframes = pickle.load(f)

    if keyframe_number < 0 or keyframe_number >= len(keyframes):
        raise IndexError(
            f"keyframe_number must be between 0 and {len(keyframes) - 1}, "
            f"but got {keyframe_number}"
        )

    keyframe = keyframes[keyframe_number]

    kf_img_bgr_original = cv2.imread(keyframe["image_path"], cv2.IMREAD_COLOR)
    test_img_bgr_original = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

    if kf_img_bgr_original is None:
        raise FileNotFoundError(f"Cannot read keyframe image: {keyframe['image_path']}")

    if test_img_bgr_original is None:
        raise FileNotFoundError(f"Cannot read test image: {test_image_path}")

    kf_img_bgr, _ = resize_keep_aspect(kf_img_bgr_original, target_width)
    test_img_bgr, _ = resize_keep_aspect(test_img_bgr_original, target_width)

    kf_kp = deserialize_keypoints(keyframe["keypoints"])
    kf_desc = keyframe["descriptors"]

    device = get_device()
    print(f"[INFO] device = {device}")

    model = load_dinov2_model(device)

    test_kp, test_desc = extract_dinov2_features_as_keypoints(
        test_img_bgr,
        model,
        device
    )


    good_matches = match_dinov2_descriptors(
        kf_desc,
        test_desc,
        ratio_thresh=ratio_thresh
    )

    H, inlier_mask, pts1, pts2 = compute_inliers_homography(
        kf_kp,
        test_kp,
        good_matches,
        ransac_thresh=ransac_thresh
    )

    num_matches = len(good_matches)

    if inlier_mask is not None:
        num_inliers = int(inlier_mask.sum())
        num_outliers = num_matches - num_inliers
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
    else:
        num_inliers = 0
        num_outliers = num_matches
        inlier_ratio = 0.0

    base_name = Path(keyframe["image_path"]).stem
    test_name = Path(test_image_path).stem
    output_file = f"dinov2_ransac_kf{keyframe_number}_{base_name}_{test_name}.png"

    vis = draw_inlier_outlier_matches(
        kf_img_bgr,
        kf_kp,
        test_img_bgr,
        test_kp,
        good_matches,
        inlier_mask,
        max_draw=max_draw
    )

    cv2.imwrite(output_file, vis)

    print("===== DINOv2 Patch Feature Comparison Result =====")
    print(f"Keyframe number        : {keyframe_number}")
    print(f"Keyframe frame_idx     : {keyframe['frame_idx']}")
    print(f"Keyframe image         : {keyframe['image_path']}")
    print(f"Test image             : {test_image_path}")
    print(f"Target width           : {target_width}")
    print(f"# keypoints keyframe   : {len(kf_kp)}")
    print(f"# keypoints test       : {len(test_kp)}")
    print(f"# good matches         : {num_matches}")
    print(f"# inliers              : {num_inliers}")
    print(f"# outliers             : {num_outliers}")
    print(f"Inlier ratio           : {inlier_ratio:.4f}")
    print(f"Visualization saved to : {output_file}")

    return {
        "keyframe_number": keyframe_number,
        "keyframe_frame_idx": keyframe["frame_idx"],
        "keyframe_image": keyframe["image_path"],
        "test_image": test_image_path,
        "num_kp_keyframe": len(kf_kp),
        "num_kp_test": len(test_kp),
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "num_outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "homography": H,
        "inlier_mask": inlier_mask,
        "output_file": output_file
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python dinov2_homography_ransac.py <test_image> <keyframe_number>")
        print("")
        print("Example:")
        print("  python dinov2_homography_ransac.py ../../Datasets/Oxford/.../1440150117948192.png 412")
        sys.exit(1)

    test_image_path = sys.argv[1]
    keyframe_number = int(sys.argv[2])

    compare_test_image_to_dinov2_keyframe(
        test_image_path=test_image_path,
        keyframe_number=keyframe_number,
        keyframe_pkl=KEYFRAME_PKL,
        target_width=DINO_IMAGE_WIDTH,
        ratio_thresh=0.80,
        ransac_thresh=8.0,
        max_draw=100
    )
