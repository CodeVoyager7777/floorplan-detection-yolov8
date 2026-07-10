"""
gn_classifier.py — Ground Truth Classifier for Door Detection

Architecture Note
-----------------
The user's high-resolution detections (e.g. 128 doors on 4959x7017 image) have tight
bounding boxes, whereas the Ground Truth annotations (993x693) have looser, wider boxes.
Standard IoU matching fails (IoU < 0.5) because the box shapes/sizes are different.

To accurately evaluate the user's actual predictions from door_locations.csv,
we transform the GT coordinates to the full-image pixel space and match them
using Center Point Distance instead of Area Overlap (IoU).
"""

import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import sys

# ─────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTION_CSV = os.path.join(PROJECT_DIR, "door_locations.csv")

IMAGES_DIR = os.path.join(
    PROJECT_DIR,
    "Floor plan object detection.yolov8",
    "train",
    "images"
)

LABELS_DIR = os.path.join(
    PROJECT_DIR,
    "Floor plan object detection.yolov8",
    "train",
    "labels"
)

# ─────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ground Truth Classifier for Door Detection"
    )
    parser.add_argument(
        "--image-path", type=str, default=None,
        help="Path to the uploaded floor plan image"
    )
    parser.add_argument(
        "--img-w", type=int, default=None,
        help="Width of the uploaded image in pixels (informational)"
    )
    parser.add_argument(
        "--img-h", type=int, default=None,
        help="Height of the uploaded image in pixels (informational)"
    )
    parser.add_argument(
        "--match-distance", type=float, default=150.0,
        help="Maximum distance in pixels between centers to consider a match (default 150)"
    )
    return parser.parse_args()

def find_matching_label(image_name):
    uploaded_stem = os.path.splitext(os.path.basename(image_name))[0]
    norm_uploaded = uploaded_stem.lower().replace(" ", "-").replace("_", "-")

    label_files = glob.glob(os.path.join(LABELS_DIR, "*.txt"))
    if not label_files:
        sys.stderr.write(
            f"The Ground Truth dataset is missing from:\\n{LABELS_DIR}\\n\\n"
            f"Note: The benchmark dataset folders were deleted to prepare the repository for GitHub deployment.\\n"
            f"To use the Ground Truth Evaluation feature again, please re-download and extract the 'Floor plan object detection.yolov8' dataset into the root directory."
        )
        sys.exit(1)

    for lf in label_files:
        lf_stem = os.path.splitext(os.path.basename(lf))[0]
        norm_lf = lf_stem.lower().replace(" ", "-").replace("_", "-")
        if norm_lf.startswith(norm_uploaded):
            return lf

    # No match found - exit cleanly with code 2 so app.py knows to hide the evaluation section
    sys.exit(2)

def find_training_image(label_file):
    stem = os.path.splitext(os.path.basename(label_file))[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(IMAGES_DIR, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No training image found for label stem '{stem}' in {IMAGES_DIR}"
    )

def main():
    args = parse_args()

    if not os.path.isfile(PREDICTION_CSV):
        print(f"[ERROR] Prediction CSV not found: {PREDICTION_CSV}")
        sys.exit(1)
        
    try:
        pred = pd.read_csv(PREDICTION_CSV)
    except pd.errors.EmptyDataError:
        # If the CSV is completely empty, treat it as 0 predictions
        pred = pd.DataFrame(columns=["image_name", "door_id", "x1", "y1", "x2", "y2", "confidence"])
    
    if "image_name" in pred.columns and not pred.empty:
        image_name = pred["image_name"].iloc[0]
    elif args.image_path:
        image_name = os.path.basename(args.image_path)
    else:
        tmp = os.path.join(PROJECT_DIR, "_uploaded_tmp.png")
        image_name = os.path.basename(tmp) if os.path.isfile(tmp) else "unknown.png"

    label_file = find_matching_label(image_name)
    train_img_path = find_training_image(label_file)
    
    print(f"[INFO] Label file : {os.path.basename(label_file)}")
    print(f"[INFO] Evaluating {len(pred)} detections from door_locations.csv")

    # Detect content region in full uploaded image
    full_img_path = args.image_path if args.image_path and os.path.isfile(args.image_path) else os.path.join(PROJECT_DIR, "_uploaded_tmp.png")
    full_img = cv2.imread(full_img_path) if os.path.isfile(full_img_path) else None
    
    def get_content_bbox(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nw = gray < 250
        rows = np.any(nw, axis=1)
        cols = np.any(nw, axis=0)
        if not rows.any() or not cols.any(): 
            return 0, 0, img.shape[1], img.shape[0]
        x_start = int(np.where(cols)[0][0])
        y_start = int(np.where(rows)[0][0])
        x_end = int(np.where(cols)[0][-1])
        y_end = int(np.where(rows)[0][-1])
        return x_start, y_start, x_end - x_start, y_end - y_start

    if full_img is not None:
        fx, fy, fw, fh = get_content_bbox(full_img)
    else:
        print("[WARNING] Could not load full image to determine scale. Assuming untransformed.")
        fx, fy, fw, fh = 0, 0, 1, 1

    # Detect content region in training image
    train_img = cv2.imread(train_img_path)
    train_h, train_w = train_img.shape[:2]
    
    if full_img is not None:
        tx, ty, tw, th = get_content_bbox(train_img)
        scale_x = fw / tw
        scale_y = fh / th
    else:
        tx, ty, tw, th = 0, 0, train_w, train_h
        scale_x, scale_y = 1.0, 1.0

    # Load GT and transform to full-image space
    gt_boxes = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            _cls, xc_n, yc_n, w_n, h_n = map(float, parts[:5])
            
            # 1. Denormalize to training image pixels
            x_train = (xc_n - w_n / 2) * train_w
            y_train = (yc_n - h_n / 2) * train_h
            w_train = w_n * train_w
            h_train = h_n * train_h
            
            # 2. Map from training content bbox to full-image content bbox
            x_full = fx + (x_train - tx) * scale_x
            y_full = fy + (y_train - ty) * scale_y
            w_full = w_train * scale_x
            h_full = h_train * scale_y
            
            gt_boxes.append({
                "x": x_full, "y": y_full, "w": w_full, "h": h_full,
                "cx": x_full + w_full/2, "cy": y_full + h_full/2
            })

    gt_df = pd.DataFrame(gt_boxes)
    print(f"[INFO] GT boxes  : {len(gt_df)}")

    # Prepare predictions
    pred["cx"] = (pred["x1"] + pred["x2"]) / 2
    pred["cy"] = (pred["y1"] + pred["y2"]) / 2
    pred["w"] = pred["x2"] - pred["x1"]
    pred["h"] = pred["y2"] - pred["y1"]
    if "confidence" not in pred.columns:
        pred["confidence"] = 1.0
        
    pred = pred.sort_values(by="confidence", ascending=False).reset_index(drop=True)

    matched_gt = set()
    tp_pred_idx = []
    detailed_results = []
    
    MATCH_DIST = args.match_distance

    for i, p in pred.iterrows():
        best_dist = float('inf')
        best_gt_idx = -1

        for j, g in gt_df.iterrows():
            if j in matched_gt:
                continue
            dist = np.sqrt((p["cx"] - g["cx"])**2 + (p["cy"] - g["cy"])**2)
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = j

        # If center is within match distance, it's a True Positive
        if best_dist < MATCH_DIST:
            tp_pred_idx.append(i)
            matched_gt.add(best_gt_idx)
            detailed_results.append([
                "TP", round(best_dist, 2),
                p["x1"], p["y1"], p["w"], p["h"], p["confidence"]
            ])
        else:
            detailed_results.append([
                "FP", round(best_dist, 2) if best_dist != float('inf') else 999.9,
                p["x1"], p["y1"], p["w"], p["h"], p["confidence"]
            ])

    tp = len(tp_pred_idx)
    fp = len(pred) - tp
    fn = len(gt_df) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    print("\n===== GN CLASSIFICATION RESULT =====")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {round(precision, 4)}")
    print(f"Recall: {round(recall, 4)}")
    print(f"F1 Score: {round(f1, 4)}")

    # Export outputs
    df_out = pd.DataFrame(detailed_results, columns=["Type", "CenterDist", "x", "y", "w", "h", "confidence"])
    df_out.to_csv(os.path.join(PROJECT_DIR, "gn_detailed_results.csv"), index=False)
    
    gt_out = gt_df.copy()
    gt_out["matched"] = [j in matched_gt for j in gt_df.index]
    gt_out.to_csv(os.path.join(PROJECT_DIR, "gn_gt_boxes.csv"), index=False)
    
    if tp_pred_idx:
        original_cols = [c for c in pred.columns if c not in ["cx", "cy", "w", "h"]]
        tp_df = pred.loc[tp_pred_idx, original_cols]
        tp_df.to_csv(os.path.join(PROJECT_DIR, "door_locations_tp.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(PROJECT_DIR, "door_locations_tp.csv"), index=False)

if __name__ == "__main__":
    main()