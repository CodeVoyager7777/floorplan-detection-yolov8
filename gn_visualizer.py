"""
gn_visualizer.py — Visualization of TP / FP / FN door detections

Fixes applied:
  - Was previously referencing `gt` and `pred` DataFrames that were never defined
    (NameError crash on first line that used them).
  - Was reading a hardcoded "input.png" that does not exist.
  - Now fully self-contained: loads all data from CSV files produced by
    gn_classifier.py.
  - Accepts optional --image-path CLI argument; falls back to _uploaded_tmp.png.

Usage:
    python gn_visualizer.py --image-path <path_to_full_floorplan_image>
    python gn_visualizer.py                     # uses _uploaded_tmp.png fallback
"""

import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd


# ─────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DETAILED_CSV = os.path.join(PROJECT_DIR, "gn_detailed_results.csv")
GT_CSV       = os.path.join(PROJECT_DIR, "gn_gt_boxes.csv")
OUTPUT_IMG   = os.path.join(PROJECT_DIR, "gn_visualization_result.png")


# ─────────────────────────────────────────
# COLORS  (BGR)
# ─────────────────────────────────────────
GREEN = (0, 220, 0)     # TP  — prediction matched GT
RED   = (0, 0, 220)     # FP  — prediction with no GT match
BLUE  = (220, 80, 0)    # FN  — GT box with no prediction match


# ─────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TP / FP / FN door detections on a floor plan image"
    )
    parser.add_argument(
        "--image-path", type=str, default=None,
        help="Path to the floor plan image to draw on (should be the full uploaded image)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def draw_box(img, x, y, w, h, color, label, thickness=4, font_scale=1.2):
    """Draw a labelled rectangle on img in-place. Coordinates are floats → cast to int."""
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img, label,
        (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    args = parse_args()

    # ── Resolve image path ───────────────────────────────────────────────────
    if args.image_path and os.path.isfile(args.image_path):
        image_path = args.image_path
    else:
        fallback = os.path.join(PROJECT_DIR, "_uploaded_tmp.png")
        if os.path.isfile(fallback):
            print(f"[INFO] --image-path not provided. Using fallback: {fallback}")
            image_path = fallback
        else:
            print(
                "[ERROR] No image found. Pass --image-path to gn_visualizer.py "
                "or run the Streamlit app first to create _uploaded_tmp.png"
            )
            sys.exit(1)

    # ── Load image ───────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    # ── Load TP / FP results (produced by gn_classifier.py) ─────────────────
    if not os.path.isfile(DETAILED_CSV):
        print(f"[ERROR] {DETAILED_CSV} not found. Run gn_classifier.py first.")
        sys.exit(1)

    detailed = pd.read_csv(DETAILED_CSV)

    # ── Load GT boxes (produced by gn_classifier.py) ─────────────────────────
    if not os.path.isfile(GT_CSV):
        print(f"[ERROR] {GT_CSV} not found. Run gn_classifier.py first.")
        sys.exit(1)

    gt_boxes = pd.read_csv(GT_CSV)

    # ── Draw TP and FP prediction boxes ─────────────────────────────────────
    for _, row in detailed.iterrows():
        box_type = row["Type"]
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        iou_score  = row.get("IoU", 0)

        if box_type == "TP":
            label = "TP"
            draw_box(img, x, y, w, h, GREEN, label)
        elif box_type == "FP":
            label = "FP"
            draw_box(img, x, y, w, h, RED, label)

    # ── Draw FN GT boxes (GT boxes that were NOT matched to any prediction) ──
    fn_count = 0
    for _, row in gt_boxes.iterrows():
        if not row.get("matched", False):
            draw_box(img, row["x"], row["y"], row["w"], row["h"], BLUE, "FN")
            fn_count += 1

    # ── Save output ──────────────────────────────────────────────────────────
    cv2.imwrite(OUTPUT_IMG, img)
    print(f"Saved: {OUTPUT_IMG}")
    print(f"  TP boxes (green): {len(detailed[detailed['Type'] == 'TP'])}")
    print(f"  FP boxes (red)  : {len(detailed[detailed['Type'] == 'FP'])}")
    print(f"  FN boxes (blue) : {fn_count}")


if __name__ == "__main__":
    main()