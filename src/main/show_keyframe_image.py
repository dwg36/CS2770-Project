import sys
import pickle
import cv2
import matplotlib.pyplot as plt


PKL_PATH = "baseline_keyframes.pkl"


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


def show_with_matplotlib(img, title, is_bgr=False):
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img, cmap=None if is_bgr else "gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_keyframe(keyframes, keyframe_idx, mode="both"):
    if keyframe_idx < 0 or keyframe_idx >= len(keyframes):
        raise IndexError(
            f"keyframe_idx must be between 0 and {len(keyframes) - 1}, "
            f"but got {keyframe_idx}"
        )

    kf = keyframes[keyframe_idx]

    img = cv2.imread(kf["image_path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {kf['image_path']}")

    keypoints = deserialize_keypoints(kf["keypoints"])

    if mode == "img":
        show_with_matplotlib(
            img,
            f"Image only | Keyframe {keyframe_idx} | frame_idx={kf['frame_idx']}",
            is_bgr=False
        )

    elif mode == "kp":
        blank = img * 0

        kp_img = cv2.drawKeypoints(
            blank,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        show_with_matplotlib(
            kp_img,
            f"Keypoints only DEFAULT | Keyframe {keyframe_idx}",
            is_bgr=True
        )

    elif mode == "kp_rich":
        blank = img * 0

        kp_img = cv2.drawKeypoints(
            blank,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        show_with_matplotlib(
            kp_img,
            f"Keypoints only RICH | Keyframe {keyframe_idx}",
            is_bgr=True
        )

    elif mode == "both":
        kp_img = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        show_with_matplotlib(
            kp_img,
            f"Image + Keypoints | Keyframe {keyframe_idx} | frame_idx={kf['frame_idx']}",
            is_bgr=True
        )

    else:
        raise ValueError("mode must be one of: img, kp, kp_rich")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 show_keyframe_image.py <keyframe_idx>")
        print("  python3 show_keyframe_image.py <keyframe_idx> img")
        print("  python3 show_keyframe_image.py <keyframe_idx> kp")
        print("  python3 show_keyframe_image.py <keyframe_idx> kp_rich")
        sys.exit(1)

    keyframe_idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) >= 3 else "both"

    with open(PKL_PATH, "rb") as f:
        keyframes = pickle.load(f)

    show_keyframe(keyframes, keyframe_idx, mode)
