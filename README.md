# Online Bangla Handwritten Text Recognition Using Stroke Trajectory Analysis

> **বাংলা হস্তলিখন পাঠ চিহ্নিতকরণ**  
> End-to-end Bengali handwriting recognition using YOLOv5 segmentation and DenseNet121 + BiGRU + CTC

---

## What This Project Does

Upload a handwritten Bengali page image and the system automatically:

1. Detects every **text line** on the page (YOLOv5)
2. Detects every **word** within each line (YOLOv5)
3. **Reads** each word using a trained deep learning OCR model (DenseNet121 + BiGRU + CTC)
4. Returns the full recognised text in a web interface where you can **correct** any word inline

---

## Project Structure

```
SECONDTYPE/
│
├── app.py                        ← Flask web server (main entry point)
├── bn_densenet_ocr.py            ← OCR model architecture + training
├── bn_grapheme.py                ← Bengali grapheme tokenizer
├── bn_banglawriting_prep.py      ← Dataset preparation script
├── requirements.txt              ← All Python dependencies
├── README.md                     ← This file
│
├── templates/
│   └── index.html                ← Web frontend (upload, results, editor)
│
├── bn_drishti_models/
│   ├── line_model_best.pt        ← Pre-trained YOLOv5 line detection model
│   └── word_model_best.pt        ← Pre-trained YOLOv5 word detection model
│
└── checkpoints_dense/
    └── best.pt                   ← Trained DenseNet OCR model (after training)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | Flask + Flask-CORS |
| Deep learning | PyTorch 2.x |
| Line & word detection | YOLOv5 (pre-trained BN-DRISHTI weights) |
| OCR backbone | DenseNet121 (ImageNet pretrained) |
| Sequence model | Bidirectional GRU (2 layers, 256 hidden) |
| Loss function | CTC — Connectionist Temporal Classification |
| Tokenizer | Unicode TR#29 Bengali Grapheme Clusters |
| Image processing | OpenCV + Pillow |
| Dataset | BanglaWriting (21,234 annotated word samples) |

---

## Installation & Setup

### Requirements
- Python 3.10 or higher
- No GPU required — CPU inference works

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
pip install yolov5
```

### Step 2 — Prepare the dataset

Download the BanglaWriting dataset, extract into `banglawriting/`, then run:

```bash
python bn_banglawriting_prep.py --data_dir banglawriting/raw --out_dir banglawriting_words
```

This reads the LabelMe JSON annotation files, crops each word bounding box from the full page images, and writes `banglawriting_words/labels.csv`.

### Step 3 — Train the OCR model

```bash
python bn_densenet_ocr.py --mode train \
    --labels banglawriting_words/labels.csv \
    --images banglawriting_words/images/ \
    --epochs 30 --batch 4
```

> Training on CPU takes approximately 8–15 hours for 30 epochs.  
> For faster training, use Google Colab with a T4 GPU (~1–2 hours).

Saves the best checkpoint to `checkpoints_dense/best.pt`.

### Step 4 — Run the web application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Usage

| Mode | Command |
|------|---------|
| Web app (full pipeline) | `python app.py` then open browser |
| Single page image (CLI) | `python app.py --page my_page.jpg` |
| Single word crop (CLI) | `python app.py --image my_word.png` |
| Debug YOLO detections | `python app.py --page my_page.jpg --debug` |

---

## Pipeline

```
Handwritten Bengali page image
              │
              ▼
    YOLOv5 — line_model_best.pt
    Detects text line bounding boxes
    Sorted top → bottom
              │
              ▼
    YOLOv5 — word_model_best.pt
    Detects word boxes within each line
    Sorted left → right
              │
              ▼
    DenseNet121 + BiGRU + CTC — best.pt
    Reads Bengali text from each word crop
              │
              ▼
    Web interface
    Annotated image · Word gallery · Editable full text
```

---

## Model Performance

Evaluated on BanglaWriting test set — 500 samples:

| Configuration | F1 | CER | WER |
|---|---|---|---|
| Full system (DenseNet + BiGRU + CTC + Grapheme tokenizer) | **0.670** | **17.8%** | **33.0%** |
| Without CLAHE preprocessing | 0.672 | 17.8% | 32.8% |
| With character tokenizer instead of grapheme | 0.408 | 35.7% | 133.4% |

---

## Dataset

**BanglaWriting** — publicly available Bengali handwritten word dataset.

- 21,234 annotated word samples
- Annotation format: LabelMe JSON with word-level bounding boxes
- Not included in this repository due to size
- Run `bn_banglawriting_prep.py` to prepare from the raw dataset

---

## References

1. QuwsarOhi / Safir et al. — *End-to-End OCR for Bengali Handwritten Words* — [GitHub](https://github.com/QuwsarOhi/bengali_word_ocr)
2. BN-DRISHTI — Bengali handwriting segmentation — crusnic-corp
3. Huang et al. — *Densely Connected Convolutional Networks* — CVPR 2017
4. Graves et al. — *Connectionist Temporal Classification* — ICML 2006
5. Cho et al. — *Learning Phrase Representations using RNN Encoder-Decoder* — EMNLP 2014
6. Kingma & Ba — *Adam: A Method for Stochastic Optimization* — ICLR 2015

---

## Note

Pre-trained model weights (`line_model_best.pt`, `word_model_best.pt`) are from the BN-DRISHTI team.  
BanglaWriting dataset belongs to its original authors.  
This project integrates both for academic purposes.
