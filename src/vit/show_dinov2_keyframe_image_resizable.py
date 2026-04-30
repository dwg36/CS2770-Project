import sys
import pickle
import cv2
import matplotlib.pyplot as plt


PKL_PATH = "dinov2_keyframes_w672.pkl"


def deserialize_keypoints(kp_list):
    keypoints = []

    for x, y, size, angle, response, octave, class_id in kp_list:
        kp = cv2.KeyPoint(
            float(x),
            float(y),
            float(size),
            float(angle),
            float(response),
            int(octave),
            int(class_id)
        )
        keypoints.append(kp)

    return keypoints


def resize_keep_aspect(image_bgr, target_width):
    if target_width is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    scale = target_width / float(w)
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image_bgr,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized


def show_with_matplotlib(img, title, is_bgr=False):
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 8))
    plt.imshow(img, cmap=None if is_bgr else "gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def print_keyframe_info(kf, keyframe_idx):
    desc = kf.get("descriptors", None)
    keypoints = kf.get("keypoints", [])

    print("===== DINOv2 Keyframe Info =====")
    print(f"keyframe_list_idx : {keyframe_idx}")
    print(f"frame_idx         : {kf.get('frame_idx')}")
    print(f"image_path        : {kf.get('image_path')}")
    print(f"original_size     : {kf.get('original_size')}")
    print(f"resized_size      : {kf.get('resized_size')}")
    print(f"resize_scale      : {kf.get('resize_scale')}")
    print(f"target_width      : {kf.get('target_width')}")
    print(f"patch_size        : {kf.get('patch_size')}")
    print(f"# keypoints       : {len(keypoints)}")

    if desc is None:
        print("descriptors       : None")
    else:
        print(f"descriptors shape : {desc.shape}")
        print(f"descriptors dtype : {desc.dtype}")


def get_display_images(kf):
    image_path = kf["image_path"]

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img_gray is None or img_color is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    target_width = kf.get("target_width", None)

    img_gray = resize_keep_aspect(img_gray, target_width)
    img_color = resize_keep_aspect(img_color, target_width)

    return img_gray, img_color


def show_keyframe(keyframes, keyframe_idx, mode="both"):
    if keyframe_idx < 0 or keyframe_idx >= len(keyframes):
        raise IndexError(
            f"keyframe_idx must be between 0 and {len(keyframes) - 1}, "
            f"but got {keyframe_idx}"
        )

    kf = keyframes[keyframe_idx]
    print_keyframe_info(kf, keyframe_idx)

    img_gray, img_color = get_display_images(kf)
    keypoints = deserialize_keypoints(kf["keypoints"])

    if mode == "img":
        show_with_matplotlib(
            img_gray,
            f"Image only | DINOv2 Keyframe {keyframe_idx} | frame_idx={kf.get('frame_idx')}",
            is_bgr=False
        )

    elif mode == "kp":
        blank = img_gray * 0

        kp_img = cv2.drawKeypoints(
            blank,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        show_with_matplotlib(
            kp_img,
            f"DINOv2 Patch Centers only | Keyframe {keyframe_idx}",
            is_bgr=True
        )

    elif mode == "kp_rich":
        blank = img_gray * 0

        kp_img = cv2.drawKeypoints(
            blank,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        show_with_matplotlib(
            kp_img,
            f"DINOv2 Patch Centers RICH | Keyframe {keyframe_idx}",
            is_bgr=True
        )

    elif mode == "both":
        kp_img = cv2.drawKeypoints(
            img_color,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        show_with_matplotlib(
            kp_img,
            f"Image + DINOv2 Patch Centers | Keyframe {keyframe_idx} | frame_idx={kf.get('frame_idx')}",
            is_bgr=True
        )

    elif mode == "grid":
        grid_img = img_color.copy()

        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(
                grid_img,
                (int(round(x)), int(round(y))),
                1,
                (0, 0, 255),
                -1
            )

        show_with_matplotlib(
            grid_img,
            f"DINOv2 Patch Grid | Keyframe {keyframe_idx} | #features={len(keypoints)}",
            is_bgr=True
        )

    else:
        raise ValueError("mode must be one of: img, kp, kp_rich, both, grid")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 show_dinov2_keyframe_image.py <keyframe_idx>")
        print("  python3 show_dinov2_keyframe_image.py <keyframe_idx> img")
        print("  python3 show_dinov2_keyframe_image.py <keyframe_idx> kp")
        print("  python3 show_dinov2_keyframe_image.py <keyframe_idx> kp_rich")
        print("  python3 show_dinov2_keyframe_image.py <keyframe_idx> grid")
        sys.exit(1)

    keyframe_idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) >= 3 else "both"

    with open(PKL_PATH, "rb") as f:
        keyframes = pickle.load(f)

    show_keyframe(keyframes, keyframe_idx, mode)
