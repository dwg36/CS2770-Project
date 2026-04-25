import cv2
import numpy as np


def extract_orb_features(image, nfeatures=2000):
    """
    Extract ORB keypoints and descriptors from a grayscale image.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio_thresh=0.75):
    """
    Match ORB descriptors using BFMatcher + KNN + Lowe's ratio test.
    ORB uses binary descriptors, so NORM_HAMMING is appropriate.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


def compute_inliers_homography(kp1, kp2, matches, ransac_thresh=3.0):
    """
    Compute inliers/outliers using Homography + RANSAC.
    Returns:
        H            : estimated homography (or None)
        inlier_mask  : mask of shape (N, 1), 1 for inlier, 0 for outlier
        pts1, pts2   : matched point arrays
    """
    if len(matches) < 4:
        return None, None, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh
    )

    return H, inlier_mask, pts1, pts2


def draw_inlier_outlier_matches(img1, kp1, img2, kp2, matches, inlier_mask,
                                max_draw=100):
    """
    Draw inlier matches in green and outlier matches in red.
    """
    if inlier_mask is None:
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:max_draw], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return vis

    inlier_mask = inlier_mask.ravel().astype(bool)

    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
    outlier_matches = [m for m, keep in zip(matches, inlier_mask) if not keep]

    inlier_matches = inlier_matches[:max_draw]
    outlier_matches = outlier_matches[:max_draw]

    vis_inliers = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        matchColor=(0, 255, 0),  # green
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    vis_outliers = cv2.drawMatches(
        img1, kp1, img2, kp2, outlier_matches, None,
        matchColor=(0, 0, 255),  # red
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Stack vertically for easier viewing
    vis = np.vstack([vis_inliers, vis_outliers])
    return vis


def compare_two_images(
    baseline_path,
    test_path,
    nfeatures=2000,
    ratio_thresh=0.75,
    ransac_thresh=3.0,
    save_vis_path="match_result.jpg",
    mask_fn=None,  # CNN: pass build_mask_fn() from cnn/mask_generator.py — TODO (ViT): swap in your equivalent here
):
    """
    Compare two images using ORB + matching + Homography RANSAC.
    """

    # 1. Load grayscale images
    img1 = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"Cannot read baseline image: {baseline_path}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read test image: {test_path}")

    # 2. Extract ORB features
    kp1, desc1 = extract_orb_features(img1, nfeatures=nfeatures)
    kp2, desc2 = extract_orb_features(img2, nfeatures=nfeatures)

    # 2.5. Strip keypoints on dynamic objects if a mask function was provided (CNN/ViT)
    if mask_fn is not None:
        import os, sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cnn'))
        from feature_filter import filter_keypoints_and_descriptors
        kp1, desc1 = filter_keypoints_and_descriptors(kp1, desc1, mask_fn(baseline_path))
        kp2, desc2 = filter_keypoints_and_descriptors(kp2, desc2, mask_fn(test_path))

    if desc1 is None or len(kp1) == 0:
        raise ValueError("No ORB features found in baseline image.")
    if desc2 is None or len(kp2) == 0:
        raise ValueError("No ORB features found in test image.")

    # 3. Descriptor matching
    good_matches = match_descriptors(desc1, desc2, ratio_thresh=ratio_thresh)

    # 4. RANSAC inlier/outlier computation
    H, inlier_mask, pts1, pts2 = compute_inliers_homography(
        kp1, kp2, good_matches, ransac_thresh=ransac_thresh
    )

    num_kp1 = len(kp1)
    num_kp2 = len(kp2)
    num_matches = len(good_matches)

    if inlier_mask is not None:
        num_inliers = int(inlier_mask.sum())
        num_outliers = num_matches - num_inliers
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
    else:
        num_inliers = 0
        num_outliers = num_matches
        inlier_ratio = 0.0

    # 5. Draw and save visualization
    vis = draw_inlier_outlier_matches(img1, kp1, img2, kp2, good_matches, inlier_mask)
    cv2.imwrite(save_vis_path, vis)

    # 6. Print summary
    print("===== Comparison Result =====")
    print(f"Baseline image: {baseline_path}")
    print(f"Test image    : {test_path}")
    print(f"# keypoints (baseline): {num_kp1}")
    print(f"# keypoints (test)    : {num_kp2}")
    print(f"# good matches        : {num_matches}")
    print(f"# inliers             : {num_inliers}")
    print(f"# outliers            : {num_outliers}")
    print(f"Inlier ratio          : {inlier_ratio:.4f}")
    print(f"Visualization saved to: {save_vis_path}")

    return {
        "num_kp_baseline": num_kp1,
        "num_kp_test": num_kp2,
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "num_outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "homography": H,
        "inlier_mask": inlier_mask
    }


#if __name__ == "__main__":
    #baseline_path = "1439388362408542_left.png"
    #test_path = "1439388362408542_right.png"
    #test_path = "1440150107635614.png"
    #test_path = "1440150117448189.png"
    #test_path = "1440150107510614.png"

#    baseline_path = "1439388386092941_left.png"
    #test_path = "1440150117760688.png"
    #test_path = "1440150117448189.png"
#    test_path = "1440150117948192.png"

#    result = compare_two_images(
#        baseline_path=baseline_path,
#        test_path=test_path,
#        nfeatures=2000,
#        ratio_thresh=0.75,
#        ransac_thresh=3.0,
#        save_vis_path="orb_match_result_007.jpg"
#    )

import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python homography_ransac.py <base_image> <test_image>")
        print("Example: python homography_ransac.py base_image/111.png test_image/222.png")
        sys.exit(1)

    baseline_path = sys.argv[1]
    test_path = sys.argv[2]

    # extract timestamp
    base_name = Path(baseline_path).stem
    test_name = Path(test_path).stem

    # output file
    output_file = f"ransac_{base_name}_{test_name}.png"

    result = compare_two_images(
        baseline_path=baseline_path,
        test_path=test_path,
        nfeatures=2000,
        ratio_thresh=0.75,
        ransac_thresh=3.0,
        save_vis_path=output_file
    )
