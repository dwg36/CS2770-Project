import csv

keyframe_info_csv = "keyframe_info_all.csv"
test_mapping_csv = "test_image_keyframe_match.csv"
output_csv = "test_image_keyframe_match_converted.csv"


# 1) Read keyframe_info_all.csv
# columns: keyframe_id, frame_idx, timestamp
frame_to_kf = {}

with open(keyframe_info_csv, "r") as f:
    reader = csv.DictReader(
        f,
        fieldnames=["keyframe_id", "frame_idx", "timestamp"]
    )

    for row in reader:
        kf_id = int(row["keyframe_id"])
        frame_idx = int(row["frame_idx"])
        frame_to_kf[frame_idx] = kf_id


# 2) Read test_image_keyframe_match.csv
# columns: test_image, center_keyframe
# Note: center_keyframe is base image frame number, not current keyframe_id
converted_rows = []

with open(test_mapping_csv, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        test_image = row["test_image"]
        frame_idx = int(row["center_keyframe"])

        if frame_idx not in frame_to_kf:
            print(f"[WARN] frame_idx {frame_idx} not found for test_image {test_image}")
            continue

        keyframe_id = frame_to_kf[frame_idx]

        converted_rows.append({
            "test_image": test_image,
            "center_keyframe": keyframe_id
        })


# 3) Save converted result
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["test_image", "center_keyframe"]
    )

    writer.writeheader()
    writer.writerows(converted_rows)


print(f"Saved: {output_csv}")
print(f"Converted rows: {len(converted_rows)}")
