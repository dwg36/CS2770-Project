import argparse
import os
import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)


DYNAMIC_PROMPT = "car . truck . bus . van . person . pedestrian . bicycle . motorcycle . cyclist ."


def boxes_cxcywh_to_xyxy(boxes, image_w, image_h):
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0] * image_w
    boxes[:, 1] = boxes[:, 1] * image_h
    boxes[:, 2] = boxes[:, 2] * image_w
    boxes[:, 3] = boxes[:, 3] * image_h

    xyxy = torch.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    xyxy[:, 0].clamp_(0, image_w - 1)
    xyxy[:, 1].clamp_(0, image_h - 1)
    xyxy[:, 2].clamp_(0, image_w - 1)
    xyxy[:, 3].clamp_(0, image_h - 1)

    return xyxy


def build_dynamic_mask_with_grounded_sam(
    image_path,
    grounding_config,
    grounding_checkpoint,
    sam_checkpoint,
    sam_model_type="vit_h",
    text_prompt=DYNAMIC_PROMPT,
    box_threshold=0.30,
    text_threshold=0.25,
    device=None
):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"[INFO] device = {device}")

    # GroundingDINO image loader
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape

    grounding_model = load_model(
        grounding_config,
        grounding_checkpoint,
        device=device
    )

    boxes, logits, phrases = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    print(f"[INFO] detected boxes = {len(boxes)}")
    for i, phrase in enumerate(phrases):
        print(f"  [{i}] {phrase}, score={logits[i].item():.3f}")

    dynamic_mask = np.zeros((h, w), dtype=np.uint8)

    if len(boxes) == 0:
        return dynamic_mask, boxes, logits, phrases, image_source

    boxes_xyxy = boxes_cxcywh_to_xyxy(boxes, w, h).cpu().numpy()

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image_source)

    for box in boxes_xyxy:
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8)

        dynamic_mask = np.maximum(dynamic_mask, best_mask)

    dynamic_mask = dynamic_mask * 255

    return dynamic_mask, boxes, logits, phrases, image_source


def draw_orb_keypoints_before_after(image_path, dynamic_mask, output_prefix, nfeatures=2000):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(image_path, cv2.IMREAD_COLOR)

    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp_all, desc_all = orb.detectAndCompute(gray, None)

    static_mask = cv2.bitwise_not(dynamic_mask)
    kp_static, desc_static = orb.detectAndCompute(gray, static_mask)

    before = cv2.drawKeypoints(
        color,
        kp_all,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    after = cv2.drawKeypoints(
        color,
        kp_static,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    cv2.imwrite(f"{output_prefix}_orb_before.png", before)
    cv2.imwrite(f"{output_prefix}_orb_after.png", after)
    cv2.imwrite(f"{output_prefix}_static_mask.png", static_mask)

    print(f"[INFO] ORB keypoints before filtering: {len(kp_all)}")
    print(f"[INFO] ORB keypoints after filtering : {len(kp_static)}")


def overlay_mask(image_path, dynamic_mask, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    overlay = image.copy()
    overlay[dynamic_mask > 0] = (0, 0, 255)

    blended = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
    cv2.imwrite(output_path, blended)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path")

    parser.add_argument(
        "--grounding-config",
        required=True,
        help="Path to GroundingDINO config.py"
    )
    parser.add_argument(
        "--grounding-checkpoint",
        required=True,
        help="Path to GroundingDINO checkpoint .pth"
    )
    parser.add_argument(
        "--sam-checkpoint",
        required=True,
        help="Path to SAM checkpoint .pth"
    )

    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output-prefix", default="dynamic_result")
    parser.add_argument("--nfeatures", type=int, default=2000)

    args = parser.parse_args()

    dynamic_mask, boxes, logits, phrases, image_source = build_dynamic_mask_with_grounded_sam(
        image_path=args.image_path,
        grounding_config=args.grounding_config,
        grounding_checkpoint=args.grounding_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    cv2.imwrite(f"{args.output_prefix}_dynamic_mask.png", dynamic_mask)
    overlay_mask(
        args.image_path,
        dynamic_mask,
        f"{args.output_prefix}_dynamic_overlay.png"
    )

    draw_orb_keypoints_before_after(
        image_path=args.image_path,
        dynamic_mask=dynamic_mask,
        output_prefix=args.output_prefix,
        nfeatures=args.nfeatures
    )

    print("[DONE]")
    print(f"saved: {args.output_prefix}_dynamic_mask.png")
    print(f"saved: {args.output_prefix}_dynamic_overlay.png")
    print(f"saved: {args.output_prefix}_orb_before.png")
    print(f"saved: {args.output_prefix}_orb_after.png")


if __name__ == "__main__":
    main()


#python make_dynamic_mask.py \
# ../../Datasets/Oxford/Oxford_Images_Validation_2015-08-21-10-40-24/rectified/left/1440150097010527.png \
# --grounding-config ../vit/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
# --grounding-checkpoint ../vit/checkpoints/groundingdino_swint_ogc.pth \
# --sam-checkpoint ../vit/checkpoints/sam_vit_h_4b8939.pth \
# --output-prefix 1440150097010527

