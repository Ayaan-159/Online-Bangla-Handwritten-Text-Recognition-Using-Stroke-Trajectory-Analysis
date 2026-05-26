# """
# bn_banglawriting_prep.py
# ────────────────────────
# Prepares the BanglaWriting dataset for training.

# BanglaWriting dataset structure (what you downloaded):
#   - BanglaWriting.zip  → contains full-page handwritten images
#   - BanglaWriting_gt.zip → contains ground truth (labels)

# This script:
#   1. Reads ground truth labels
#   2. Copies / organises word images into banglawriting_words/images/
#   3. Writes banglawriting_words/labels.csv  (filename, text)

# Usage
# ─────
#   python bn_banglawriting_prep.py --data_dir .  --out_dir banglawriting_words

#   --data_dir   folder where you extracted the two ZIPs  (default: current dir)
#   --out_dir    where to write images/ and labels.csv     (default: banglawriting_words)

# The script auto-detects common BanglaWriting folder layouts.
# """

# import os
# import csv
# import shutil
# import argparse
# from pathlib import Path


# def find_images_and_labels(data_dir: Path):
#     """
#     Auto-detect the BanglaWriting folder layout.

#     Common layouts after extraction:
#       Layout A:
#         data_dir/
#           BanglaWriting/
#             data/
#               <writer_id>/
#                 <word_id>.png
#             gt.txt  or  labels.csv
#       Layout B:
#         data_dir/
#           images/  (already extracted word crops)
#           labels.csv

#     Returns (image_root, label_file_path)
#     """
#     candidates_label = list(data_dir.rglob('gt.txt')) + \
#                        list(data_dir.rglob('labels.csv')) + \
#                        list(data_dir.rglob('ground_truth.txt')) + \
#                        list(data_dir.rglob('*.csv'))

#     candidates_img = list(data_dir.rglob('images'))
#     candidates_img += list(data_dir.rglob('data'))

#     label_file = None
#     for c in candidates_label:
#         if c.stat().st_size > 100:
#             label_file = c
#             break

#     img_root = None
#     for c in candidates_img:
#         if c.is_dir() and any(c.rglob('*.png')):
#             img_root = c
#             break
#     if img_root is None:
#         # Maybe images are flat in data_dir
#         if any(data_dir.rglob('*.png')):
#             img_root = data_dir

#     return img_root, label_file


# def parse_label_file(label_file: Path):
#     """
#     Parse a label file into list of (filename, text) tuples.
#     Handles CSV (comma/tab separated) and space-separated formats.
#     """
#     entries = []
#     with open(label_file, encoding='utf-8', errors='replace') as f:
#         content = f.read()

#     lines = content.splitlines()

#     for line in lines:
#         line = line.strip()
#         if not line or line.startswith('#'):
#             continue

#         # Try comma-separated
#         if ',' in line:
#             parts = line.split(',', 1)
#         # Try tab-separated
#         elif '\t' in line:
#             parts = line.split('\t', 1)
#         # Try space-separated (filename.png বাংলা)
#         else:
#             parts = line.split(None, 1)

#         if len(parts) == 2:
#             fname, text = parts[0].strip(), parts[1].strip()
#             if fname and text:
#                 entries.append((fname, text))

#     return entries


# def prep(data_dir: Path, out_dir: Path):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     img_out = out_dir / 'images'
#     img_out.mkdir(exist_ok=True)
#     csv_out = out_dir / 'labels.csv'

#     print(f"Searching for dataset in: {data_dir}")
#     img_root, label_file = find_images_and_labels(data_dir)

#     if label_file is None:
#         print("\n[ERROR] Could not find a label file.")
#         print("  Expected: gt.txt, labels.csv, or ground_truth.txt")
#         print("  Please check the contents of your extracted ZIPs.")
#         return

#     if img_root is None:
#         print("\n[ERROR] Could not find image folder with .png files.")
#         return

#     print(f"  Label file : {label_file}")
#     print(f"  Image root : {img_root}")

#     entries = parse_label_file(label_file)
#     print(f"  Found {len(entries)} label entries")

#     # Build an index of all available images for fast lookup
#     all_images = {}
#     for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
#         for p in img_root.rglob(ext):
#             all_images[p.name] = p
#             all_images[p.stem] = p   # also index by stem

#     written = 0
#     skipped = 0
#     rows = []

#     for fname, text in entries:
#         # Normalise filename
#         fpath_name = Path(fname).name
#         fpath_stem = Path(fname).stem

#         src = all_images.get(fpath_name) or all_images.get(fpath_stem)

#         if src is None:
#             skipped += 1
#             continue

#         dest_name = fpath_name if '.' in fpath_name else fpath_name + '.png'
#         dest = img_out / dest_name

#         if not dest.exists():
#             shutil.copy2(src, dest)

#         rows.append((dest_name, text))
#         written += 1

#     # Write CSV
#     with open(csv_out, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['filename', 'text'])
#         writer.writerows(rows)

#     print(f"\n  ✓ Written : {written} samples → {csv_out}")
#     if skipped:
#         print(f"  ⚠ Skipped : {skipped} entries (image not found)")
#     print(f"\nNext step:")
#     print(f"  python bn_densenet_ocr.py --mode train \\")
#     print(f"      --labels {csv_out} \\")
#     print(f"      --images {img_out} \\")
#     print(f"      --epochs 30 --batch 8")


# if __name__ == '__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('--data_dir', default='.',
#                    help='Folder where you extracted the BanglaWriting ZIPs')
#     p.add_argument('--out_dir', default='banglawriting_words',
#                    help='Output folder for prepared dataset')
#     args = p.parse_args()

#     prep(Path(args.data_dir).resolve(), Path(args.out_dir).resolve())


"""
bn_banglawriting_prep.py
────────────────────────
Prepares the BanglaWriting dataset from LabelMe-format JSON annotations.

Expected input structure (what you have):
  data_dir/
    ├── image1.jpg          ← original handwritten page image
    ├── image1.json         ← LabelMe annotation for image1.jpg
    ├── image2.jpg
    ├── image2.json
    └── ...

Each JSON looks like:
  {
    "shapes": [
      {
        "label": "বৃহত্তর",
        "points": [[x1, y1], [x2, y2]]   ← bounding box (top-left, bottom-right)
      },
      ...
    ]
  }

What this script does:
  1. Finds every JSON file
  2. Loads the matching image
  3. Crops each word bounding box -> saves as individual PNG
  4. Writes banglawriting_words/labels.csv

Usage
-----
  python bn_banglawriting_prep.py --data_dir . --out_dir banglawriting_words

  --data_dir   folder containing your images + JSON files  (default: .)
  --out_dir    where to write cropped images + labels.csv  (default: banglawriting_words)
  --padding    extra pixels around each crop (default: 4)
"""

import os
import csv
import json
import argparse
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
PADDING  = 4   # pixels of padding around each crop


def find_image_for_json(json_path: Path):
    """
    Given a JSON file, find the matching image.
    Tries same stem with common image extensions.
    Also checks if 'imagePath' key is set inside the JSON.
    """
    # Check inside JSON for imagePath field (LabelMe stores this)
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        if 'imagePath' in data:
            candidate = json_path.parent / data['imagePath']
            if candidate.exists():
                return candidate
    except Exception:
        pass

    # Try same stem + image extension
    for ext in IMG_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


def crop_word(img: np.ndarray, points: list, padding: int = PADDING):
    """
    Crop a word from the image using bounding box points.

    points is a list of [x, y] coordinates.
    Works with both 2-point (rectangle) and polygon (takes bounding rect).
    """
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(img.shape[1], int(max(xs)) + padding)
    y2 = min(img.shape[0], int(max(ys)) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def prep(data_dir: Path, out_dir: Path, padding: int = PADDING):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / 'images'
    img_out.mkdir(exist_ok=True)
    csv_out = out_dir / 'labels.csv'

    # Find all JSON files recursively
    json_files = sorted(data_dir.rglob('*.json'))

    if not json_files:
        print(f"[ERROR] No JSON files found in: {data_dir}")
        return

    print(f"Found {len(json_files)} JSON files in: {data_dir}")
    print(f"Output folder: {out_dir}\n")

    rows        = []
    total_crops = 0
    skipped     = 0
    no_image    = 0

    for json_path in json_files:
        # Load JSON
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [SKIP] Cannot read {json_path.name}: {e}")
            skipped += 1
            continue

        shapes = data.get('shapes', [])
        if not shapes:
            continue

        # Find matching image
        img_path = find_image_for_json(json_path)
        if img_path is None:
            print(f"  [SKIP] No image found for {json_path.name}")
            no_image += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot read image {img_path.name}")
            no_image += 1
            continue

        # Crop each shape
        page_name = json_path.stem   # e.g. "page_001"
        for idx, shape in enumerate(shapes):
            label  = shape.get('label', '').strip()
            points = shape.get('points', [])

            if not label or not points:
                skipped += 1
                continue

            crop = crop_word(img, points, padding)
            if crop is None:
                skipped += 1
                continue

            # Save crop
            out_name = f"{page_name}_{idx:04d}.png"
            out_path = img_out / out_name
            cv2.imwrite(str(out_path), crop)

            rows.append((out_name, label))
            total_crops += 1

        print(f"  OK {json_path.name:40s} -> {len(shapes)} words")

    # Write CSV
    with open(csv_out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'text'])
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"  Total word crops saved : {total_crops}")
    print(f"  Skipped (bad data)     : {skipped}")
    print(f"  Missing images         : {no_image}")
    print(f"  CSV written to         : {csv_out}")
    print(f"\nNext step -- train the model:")
    print(f"  python bn_densenet_ocr.py --mode train \\")
    print(f"      --labels {csv_out} \\")
    print(f"      --images {img_out} \\")
    print(f"      --epochs 30 --batch 4")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='.',
                   help='Folder containing images + JSON files')
    p.add_argument('--out_dir',  default='banglawriting_words',
                   help='Output folder for cropped images + labels.csv')
    p.add_argument('--padding',  type=int, default=4,
                   help='Extra pixels of padding around each crop (default: 4)')
    args = p.parse_args()

    prep(Path(args.data_dir).resolve(),
         Path(args.out_dir).resolve(),
         args.padding)