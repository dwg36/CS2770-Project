import pickle
import cv2
import torch
import numpy as np


PATCH_SIZE = 14

# Original size, if None
DINO_IMAGE_WIDTH = 672


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


def serialize_keypoints(keypoints):
    return [
        (
            kp.pt[0],
            kp.pt[1],
            kp.size,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id
        )
        for kp in keypoints
    ]


def build_dinov2_keyframes(
    orb_keyframe_path="baseline_keyframes.pkl",
    output_path="dinov2_keyframes_w672.pkl",
    target_width=DINO_IMAGE_WIDTH
):
    device = get_device()
    print(f"[INFO] device = {device}")
    print(f"[INFO] target_width = {target_width}")

    model = load_dinov2_model(device)

    with open(orb_keyframe_path, "rb") as f:
        orb_keyframes = pickle.load(f)

    dinov2_keyframes = []

    for i, kf in enumerate(orb_keyframes):
        image_path = kf["image_path"]

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Cannot read keyframe image: {image_path}")
            continue

        orig_h, orig_w = image_bgr.shape[:2]

        resized_bgr, scale = resize_keep_aspect(
            image_bgr,
            target_width
        )

        resized_h, resized_w = resized_bgr.shape[:2]

        kp, desc = extract_dinov2_features_as_keypoints(
            resized_bgr,
            model,
            device
        )

        dinov2_keyframes.append({
            "frame_idx": kf["frame_idx"],
            "image_path": image_path,
            "original_size": (orig_w, orig_h),
            "resized_size": (resized_w, resized_h),
            "resize_scale": scale,
            "target_width": target_width,
            "patch_size": PATCH_SIZE,
            "keypoints": serialize_keypoints(kp),
            "descriptors": desc
        })

        print(
            f"[KF] {i + 1}/{len(orb_keyframes)} | "
            f"frame_idx={kf['frame_idx']} | "
            f"orig={orig_w}x{orig_h} | "
            f"resized={resized_w}x{resized_h} | "
            f"features={len(kp)}"
        )

    with open(output_path, "wb") as f:
        pickle.dump(dinov2_keyframes, f)

    print("===== DINOv2 Keyframe Build Done =====")
    print(f"# keyframes : {len(dinov2_keyframes)}")
    print(f"saved to    : {output_path}")


if __name__ == "__main__":
    build_dinov2_keyframes(
        orb_keyframe_path="baseline_keyframes.pkl",
        output_path="dinov2_keyframes_w672.pkl",
        target_width=DINO_IMAGE_WIDTH
    )
