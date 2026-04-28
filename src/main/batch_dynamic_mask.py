import argparse
import os
import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict
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


def build_dynamic_mask(image_path, grounding_model, sam, device,
                       text_prompt, box_threshold, text_threshold):

    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape

    boxes, logits, phrases = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    dynamic_mask = np.zeros((h, w), dtype=np.uint8)

    if len(boxes) == 0:
        return dynamic_mask

    boxes_xyxy = boxes_cxcywh_to_xyxy(boxes, w, h).cpu().numpy()

    predictor = SamPredictor(sam)
    predictor.set_image(image_source)

    for box in boxes_xyxy:
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )

        best_mask = masks[np.argmax(scores)].astype(np.uint8)
        dynamic_mask = np.maximum(dynamic_mask, best_mask)

    return dynamic_mask * 255


def overlay_mask(image_path, dynamic_mask, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    overlay = image.copy()
    overlay[dynamic_mask > 0] = (0, 0, 255)

    blended = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
    cv2.imwrite(output_path, blended)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--grounding-config", required=True)
    parser.add_argument("--grounding-checkpoint", required=True)
    parser.add_argument("--sam-checkpoint", required=True)

    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"[INFO] device = {device}")

    # Load model only once (for execution speed)
    grounding_model = load_model(
        args.grounding_config,
        args.grounding_checkpoint,
        device=device
    )

    sam = sam_model_registry[args.sam_model_type](
        checkpoint=args.sam_checkpoint
    )
    sam.to(device=device)

    # Iterative image sequence
    image_files = sorted(os.listdir(args.input_dir))

    for fname in image_files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(args.input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        print(f"\n[PROCESS] {fname}")

        dynamic_mask = build_dynamic_mask(
            input_path,
            grounding_model,
            sam,
            device,
            DYNAMIC_PROMPT,
            args.box_threshold,
            args.text_threshold
        )

        mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
        overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")

        cv2.imwrite(mask_path, dynamic_mask)
        overlay_mask(input_path, dynamic_mask, overlay_path)

        print(f"  saved: {mask_path}")
        print(f"  saved: {overlay_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()

#python batch_dynamic_mask.py \
# --input-dir ~/Project/dataset/validation \
# --output-dir ~/Project/dataset/validation_result_sam_dino \
# --grounding-config ~/GroundingDINO_src_backup/groundingdino/config/GroundingDINO_SwinT_OGC.py \
# --grounding-checkpoint ~/weights/groundingdino_swint_ogc.pth \
# --sam-checkpoint ~/weights/sam_vit_b_01ec64.pth \
# --sam-model-type vit_b
