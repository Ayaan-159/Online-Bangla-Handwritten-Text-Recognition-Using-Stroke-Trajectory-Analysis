

import os, sys, time, base64, threading, warnings
import numpy as np
import cv2
import torch
from io import BytesIO
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).parent
MODELS_DIR      = SCRIPT_DIR / "bn_drishti_models"
LINE_MODEL_PATH = MODELS_DIR / "line_model_best.pt"
WORD_MODEL_PATH = MODELS_DIR / "word_model_best.pt"
OCR_MODEL_PATH  = SCRIPT_DIR / "checkpoints_dense" / "best.pt"

IMG_SIZE  = 1280
LINE_CONF = 0.25
WORD_CONF = 0.20   # slightly lower to catch more words
PADDING   = 8

LINE_COLORS_BGR = [
    (252,  92, 124), ( 56, 189, 248), ( 34, 197,  94),
    (245, 158,  11), (168,  85, 247), (251, 146,  60),
    (236,  72, 153), ( 20, 184, 166), ( 99, 102, 241),
    (239,  68,  68), (250, 204,  21), ( 52, 211, 153),
]

# ── Global model cache ────────────────────────────────────────────────────────
_line_model  = None
_word_model  = None
_ocr_model   = None
_models_lock = threading.Lock()


def _load_yolo(path: Path):
    import yolov5
    orig = torch.load
    torch.load = lambda *a, **kw: orig(*a, **{**kw, 'weights_only': False})
    try:
        model = yolov5.load(str(path))
    finally:
        torch.load = orig
    return model


def load_models():
    global _line_model, _word_model, _ocr_model
    if _line_model is not None:
        return

    print("\n── Loading models ───────────────────────────────────────")
    for path, name in [(LINE_MODEL_PATH, "line"), (WORD_MODEL_PATH, "word")]:
        if not path.exists():
            print(f"[ERROR] {name} model not found: {path}")
            sys.exit(1)

    print("  Loading line model …")
    _line_model = _load_yolo(LINE_MODEL_PATH)
    _line_model.conf = LINE_CONF
    _line_model.iou  = 0.45

    print("  Loading word model …")
    _word_model = _load_yolo(WORD_MODEL_PATH)
    _word_model.conf = WORD_CONF
    _word_model.iou  = 0.45

    if OCR_MODEL_PATH.exists():
        print("  Loading DenseNet OCR …")
        from bn_densenet_ocr import BengaliDenseNetOCR
        _ocr_model = BengaliDenseNetOCR(str(OCR_MODEL_PATH))
    else:
        print(f"  [WARN] OCR model not found at {OCR_MODEL_PATH} — text will be empty")
        _ocr_model = None

    print("  All models ready.\n")


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image(data: bytes):
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_rgb  = np.array(pil_img)
    np_bgr  = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    return pil_img, np_bgr


def ndarray_to_b64(img_bgr: np.ndarray, fmt=".jpg") -> str:
    ok, buf = cv2.imencode(fmt, img_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, 90] if fmt == ".jpg" else [])
    if not ok:
        return ""
    b64  = base64.b64encode(buf.tobytes()).decode()
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_paragraph_blocks(pil_img) -> list:
    """Run line YOLO — returns raw paragraph-level blocks."""
    results    = _line_model(pil_img, size=IMG_SIZE)
    dets       = results.xyxy[0].cpu().numpy()
    W, H       = pil_img.size
    blocks     = []
    for det in dets:
        x1, y1, x2, y2, conf, _ = det
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 > x1 and y2 > y1:
            blocks.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":round(float(conf),4)})
    blocks.sort(key=lambda b: b["y1"])
    return blocks


def detect_all_words(np_bgr: np.ndarray) -> list:
    """
    Run word YOLO on the FULL image at once.
    Much more accurate than running per-line-crop because the model
    sees full context and we avoid coordinate offset bugs.
    """
    H, W = np_bgr.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB))
    results = _word_model(pil, size=IMG_SIZE)
    dets    = results.xyxy[0].cpu().numpy()

    words = []
    for det in dets:
        x1, y1, x2, y2, conf, _ = det
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 > x1 and y2 > y1:
            cy = (y1 + y2) / 2
            words.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                          "conf":round(float(conf),4), "cy":cy})
    words.sort(key=lambda w: (w["y1"], w["x1"]))
    return words


def cluster_words_into_lines(words: list, gap_factor: float = 0.6) -> list:
    """
    Group word bboxes into text lines by vertical proximity.

    Algorithm:
    1. Sort words by centroid-y
    2. Compute median word height → use as line-gap threshold
    3. Greedy scan: if next word's cy is within threshold of current group's
       mean cy → same line, else new line

    This correctly splits paragraph blocks into individual text lines
    even when the YOLO line model over-groups them.
    """
    if not words:
        return []

    heights  = [w["y2"] - w["y1"] for w in words]
    med_h    = float(np.median(heights)) if heights else 20.0
    gap      = med_h * gap_factor

    # Sort by cy
    sorted_w = sorted(words, key=lambda w: w["cy"])
    lines    = [[sorted_w[0]]]

    for word in sorted_w[1:]:
        last_line  = lines[-1]
        last_cy    = np.mean([w["cy"] for w in last_line])
        if abs(word["cy"] - last_cy) <= gap:
            last_line.append(word)
        else:
            lines.append([word])

    # Sort each line left→right, build line dicts
    result = []
    for i, line_words in enumerate(lines):
        line_words.sort(key=lambda w: w["x1"])
        xs  = [w["x1"] for w in line_words]
        ys  = [w["y1"] for w in line_words]
        x2s = [w["x2"] for w in line_words]
        y2s = [w["y2"] for w in line_words]
        for j, w in enumerate(line_words):
            w["word_id"] = j + 1
        result.append({
            "line_id": i + 1,
            "x1": min(xs), "y1": min(ys),
            "x2": max(x2s), "y2": max(y2s),
            "conf": round(float(np.mean([w["conf"] for w in line_words])), 4),
            "words": line_words,
        })

    return result


def filter_lines_by_blocks(lines: list, blocks: list,
                            iou_thresh: float = 0.1) -> list:
    """
    Optional: keep only lines that overlap with a detected paragraph block.
    This removes noise words outside the main text area.
    If no blocks detected, return all lines unchanged.
    """
    if not blocks:
        return lines

    def overlap_y(line, block):
        inter = max(0, min(line["y2"], block["y2"]) - max(line["y1"], block["y1"]))
        line_h = max(1, line["y2"] - line["y1"])
        return inter / line_h

    filtered = []
    for line in lines:
        for block in blocks:
            if overlap_y(line, block) > iou_thresh:
                filtered.append(line)
                break
    return filtered if filtered else lines


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_annotations(np_bgr: np.ndarray, lines_data: list) -> np.ndarray:
    """
    Draw line boxes and word boxes with:
    - Each line gets a unique color
    - Line label at top-left of the line box
    - Word boxes in white with thin border
    - Recognised text above each word box (if available)
    """
    vis = np_bgr.copy()

    for line in lines_data:
        lid   = line["line_id"]
        color = LINE_COLORS_BGR[(lid - 1) % len(LINE_COLORS_BGR)]

        # Line bounding box (thick, colored)
        cv2.rectangle(vis,
                      (line["x1"], line["y1"]),
                      (line["x2"], line["y2"]),
                      color, 2)

        # Line label background
        label    = f"L{lid}"
        lbl_x    = line["x1"]
        lbl_y    = line["y1"] - 6 if line["y1"] > 20 else line["y2"] + 16
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (lbl_x - 1, lbl_y - th - 2),
                      (lbl_x + tw + 2, lbl_y + 2), color, -1)
        cv2.putText(vis, label, (lbl_x, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Word boxes
        for word in line.get("words", []):
            cv2.rectangle(vis,
                          (word["x1"], word["y1"]),
                          (word["x2"], word["y2"]),
                          color, 1)

            # Show recognised text above word box
            text = word.get("text", "")
            if text:
                tx = word["x1"]
                ty = max(12, word["y1"] - 3)
                (txw, txh), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.rectangle(vis, (tx - 1, ty - txh - 1),
                              (tx + txw + 1, ty + 2),
                              (255, 255, 255), -1)
                cv2.putText(vis, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            color, 1, cv2.LINE_AA)

    return vis


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "segmentation_ready": _line_model is not None,
        "ocr_ready":          _ocr_model is not None,
    })


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image field"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}:
        return jsonify({"error": f"Unsupported: {ext}"}), 400

    with _models_lock:
        load_models()

    t0 = time.time()
    img_bytes = file.read()
    try:
        pil_img, np_bgr = load_image(img_bytes)
    except Exception as e:
        return jsonify({"error": f"Bad image: {e}"}), 400

    # ── Step 1: detect paragraph blocks (line YOLO) ───────────────────
    try:
        blocks = detect_paragraph_blocks(pil_img)
    except Exception as e:
        return jsonify({"error": f"Line detection failed: {e}"}), 500

    # ── Step 2: detect all words on full image ────────────────────────
    try:
        all_words = detect_all_words(np_bgr)
    except Exception as e:
        return jsonify({"error": f"Word detection failed: {e}"}), 500

    if not all_words:
        return jsonify({
            "annotated_image": ndarray_to_b64(np_bgr),
            "lines": [], "full_text": "",
            "stats": {"lines":0,"words":0,"elapsed_s":round(time.time()-t0,2)},
            "warning": "No words detected.",
        })

    # ── Step 3: cluster words into true text lines ────────────────────
    lines_data = cluster_words_into_lines(all_words, gap_factor=0.6)
    lines_data = filter_lines_by_blocks(lines_data, blocks)

    # ── Step 4: OCR each word ─────────────────────────────────────────
    full_text_parts = []

    for line in lines_data:
        words = line["words"]
        line_text_parts = []

        if _ocr_model and words:
            crops = []
            for w in words:
                crop = np_bgr[w["y1"]:w["y2"], w["x1"]:w["x2"]]
                crops.append(crop if crop.size > 0
                              else np.ones((32, 64, 3), np.uint8) * 255)
            texts = _ocr_model.recognize_batch(crops)
            for w, text in zip(words, texts):
                w["text"] = text
                if text:
                    line_text_parts.append(text)
        else:
            for w in words:
                w["text"] = ""

        line["line_text"] = " ".join(line_text_parts)
        line["word_count"] = len(words)
        if line["line_text"]:
            full_text_parts.append(line["line_text"])

    full_text  = "\n".join(full_text_parts)
    annotated  = draw_annotations(np_bgr, lines_data)
    ann_b64    = ndarray_to_b64(annotated)
    elapsed    = round(time.time() - t0, 2)
    total_words = sum(l["word_count"] for l in lines_data)

    # ── Build JSON response ───────────────────────────────────────────
    response_lines = []
    for line in lines_data:
        response_words = []
        for w in line["words"]:
            crop = np_bgr[w["y1"]:w["y2"], w["x1"]:w["x2"]]
            response_words.append({
                "word_id":    w["word_id"],
                "x1": w["x1"], "y1": w["y1"],
                "x2": w["x2"], "y2": w["y2"],
                "conf":       w["conf"],
                "text":       w.get("text", ""),
                "crop_image": ndarray_to_b64(crop) if crop.size > 0 else "",
            })
        response_lines.append({
            "line_id":   line["line_id"],
            "conf":      line["conf"],
            "x1": line["x1"], "y1": line["y1"],
            "x2": line["x2"], "y2": line["y2"],
            "line_text": line["line_text"],
            "word_count": line["word_count"],
            "words":     response_words,
        })

    return jsonify({
        "annotated_image": ann_b64,
        "lines":           response_lines,
        "full_text":       full_text,
        "stats": {
            "lines":     len(lines_data),
            "words":     total_words,
            "elapsed_s": elapsed,
        },
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  BN-DRISHTI  Bengali Handwriting Recognition  v3")
    print("  http://localhost:5000")
    print("=" * 60)
    with _models_lock:
        load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)