"""
bn_densenet_ocr.py
───────────────────
Clean PyTorch reimplementation of the QuwsarOhi / Safir et al. (2021) approach:
  DenseNet121 (ImageNet pretrained) → column-wise GRU → CTC loss

Paper: "End-to-End Optical Character Recognition for Bengali Handwritten Words"
       CER=9.1%, WER=27.3% on BanglaWriting dataset

Key insight vs our earlier CNN:
  • DenseNet121 is pretrained on 1.2M ImageNet images → rich visual features
    even before seeing a single Bengali word
  • Transfer learning means CTC alignment is the only thing to learn,
    not low-level features AND alignment simultaneously
  • This is why it works with ~21K samples where a scratch CNN fails

Usage
─────
  # Step 1: prepare BanglaWriting word crops (run bn_banglawriting_prep.py first)
  python bn_densenet_ocr.py --mode train \
      --labels banglawriting_words/labels.csv \
      --images banglawriting_words/images/ \
      --epochs 50

  # Step 2: predict on your word crops
  python bn_densenet_ocr.py --mode predict \
      --model checkpoints_dense/best.pt \
      --words output/words/

  # Step 3: plug into app.py (see bottom of this file)
"""

from __future__ import annotations
import os, sys, csv, json, argparse, time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image as PILImage

from bn_grapheme import BnGraphemeTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

# DenseNet121 expects 224×224 RGB normalised to ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_H = 64    # fixed height for word images before feeding to DenseNet
IMG_W = 256   # fixed width  (pad/truncate)


def preprocess_word(img_bgr: np.ndarray,
                    augment: bool = False) -> torch.Tensor:
    """
    BGR word crop → normalised tensor [3, IMG_H, IMG_W].

    Steps
    ─────
    1. Grayscale + auto-invert (ensure ink dark on white)
    2. CLAHE contrast enhancement
    3. Resize height to IMG_H, pad/truncate width to IMG_W
    4. Convert to 3-channel (DenseNet expects RGB)
    5. ImageNet normalisation
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 127:
        gray = 255 - gray

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # Optional augmentation
    if augment:
        # Slight random rotation
        h, w = gray.shape
        angle = np.random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # Random brightness
        alpha = np.random.uniform(0.85, 1.15)
        gray  = np.clip(gray.astype(float) * alpha, 0, 255).astype(np.uint8)

    # Resize height, scale width proportionally
    h, w = gray.shape
    if h == 0 or w == 0:
        return torch.zeros(3, IMG_H, IMG_W)
    scale  = IMG_H / h
    new_w  = max(1, int(w * scale))
    resized = cv2.resize(gray, (new_w, IMG_H), interpolation=cv2.INTER_AREA)

    # Pad or truncate width
    if new_w < IMG_W:
        pad    = np.full((IMG_H, IMG_W - new_w), 255, dtype=np.uint8)
        canvas = np.hstack([resized, pad])
    else:
        canvas = resized[:, :IMG_W]

    # → 3-channel RGB
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # Tensor + ImageNet normalise
    t = transforms.functional.to_tensor(PILImage.fromarray(rgb))
    t = transforms.functional.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class BanglaWordDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str,
                 tok: BnGraphemeTokenizer, augment: bool = False):
        self.tok     = tok
        self.augment = augment
        self.entries: List[Tuple[str, str]] = []

        image_dir = Path(image_dir)
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))

        start = 0
        if rows and rows[0][0].strip().lower() in ("filename","file","image","path"):
            start = 1

        for row in rows[start:]:
            if len(row) < 2: continue
            fname, text = row[0].strip(), row[1].strip()
            if not text: continue
            p = image_dir / fname
            if p.exists():
                self.entries.append((str(p), text))

        print(f"[Dataset] {len(self.entries)} samples from {csv_path}")

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        path, text = self.entries[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.ones((48, 100, 3), np.uint8) * 255
        tensor = preprocess_word(img, augment=self.augment)
        ids    = self.tok.encode(text)
        ids    = [i for i in ids if i not in (self.tok.PAD, self.tok.BOS, self.tok.EOS)]
        if not ids: ids = [self.tok.UNK]
        return tensor, torch.tensor(ids, dtype=torch.long)


def ctc_collate(batch):
    imgs, labels = zip(*batch)
    imgs         = torch.stack(imgs)
    label_lens   = torch.tensor([l.size(0) for l in labels], dtype=torch.long)
    labels_flat  = torch.cat(list(labels))
    return imgs, labels_flat, label_lens


# ──────────────────────────────────────────────────────────────────────────────
# Model: DenseNet121 encoder + GRU + CTC
# ──────────────────────────────────────────────────────────────────────────────

class DenseNetGRU_OCR(nn.Module):
    """
    DenseNet121 feature extractor with ImageNet weights,
    followed by a bidirectional GRU sequence model and CTC output.

    Input:  [B, 3, H, W]   (H=64, W=256)
    Output: [T, B, vocab]  log-softmax, T = W // 32 = 8

    The DenseNet features are extracted from the last dense block
    (before the global pool), giving spatial feature maps that we
    read column-by-column as a time sequence.
    """

    def __init__(self, vocab_size: int,
                 gru_hidden:  int = 256,
                 gru_layers:  int = 2,
                 dropout:     float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size

        # ── DenseNet121 feature extractor ─────────────────────────────────
        densenet    = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # Remove classifier and global pool — keep only the feature layers
        # DenseNet features output: [B, 1024, H/32, W/32]
        self.features = densenet.features   # stops at final BatchNorm
        self.feat_dim = 1024                # DenseNet121 output channels

        # Collapse height dimension to 1 (adaptive pool)
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))  # [B, 1024, 1, T]

        # Optional channel reduction before GRU
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, gru_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Bidirectional GRU ─────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size    = gru_hidden,
            hidden_size   = gru_hidden,
            num_layers    = gru_layers,
            bidirectional = True,
            dropout       = dropout if gru_layers > 1 else 0.0,
            batch_first   = False,   # time-first
        )

        # ── Output layer ──────────────────────────────────────────────────
        self.fc = nn.Linear(gru_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DenseNet feature extraction
        f = self.features(x)                 # [B, 1024, H/32, W/32]
        f = F.relu(f, inplace=True)
        f = self.h_pool(f)                   # [B, 1024, 1, T]
        f = f.squeeze(2)                     # [B, 1024, T]
        f = f.permute(2, 0, 1)              # [T, B, 1024]

        # Project
        f = self.proj(f)                     # [T, B, gru_hidden]

        # GRU
        f, _ = self.gru(f)                  # [T, B, gru_hidden*2]

        # Output
        logits    = self.fc(f)              # [T, B, vocab]
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_densenet(self, freeze: bool = True):
        """
        Freeze/unfreeze DenseNet feature layers.
        Phase 1: freeze DenseNet, only train GRU+FC (fast convergence)
        Phase 2: unfreeze everything for fine-tuning
        """
        for param in self.features.parameters():
            param.requires_grad = not freeze


# ──────────────────────────────────────────────────────────────────────────────
# CTC Greedy Decoder
# ──────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int) -> List[List[int]]:
    indices = log_probs.argmax(dim=2).permute(1, 0)  # [B, T]
    results = []
    for seq in indices:
        decoded, prev = [], None
        for tok in seq.tolist():
            if tok != blank_id and tok != prev:
                decoded.append(tok)
            prev = tok
        results.append(decoded)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def cer(pred: str, ref: str) -> float:
    if not ref: return 0.0 if not pred else 1.0
    n, m = len(ref), len(pred)
    dp = list(range(n+1))
    for j in range(1, m+1):
        prev = dp[0]; dp[0] = j
        for i in range(1, n+1):
            tmp = dp[i]
            dp[i] = prev if pred[j-1]==ref[i-1] else 1+min(prev,dp[i],dp[i-1])
            prev = tmp
    return min(dp[n]/len(ref), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  DenseNet121 + GRU + CTC  —  Bengali OCR")
    print(f"  Device : {device.upper()}")
    print(f"  Labels : {args.labels}")
    print(f"{'='*60}\n")

    os.makedirs(args.out, exist_ok=True)
    tok = BnGraphemeTokenizer()

    # Dataset
    full_ds = BanglaWordDataset(args.labels, args.images, tok, augment=False)
    n = len(full_ds)
    if n == 0:
        print("ERROR: no samples loaded"); sys.exit(1)

    n_val   = max(1, int(n * 0.15))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds.dataset.augment = True

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                               shuffle=True,  collate_fn=ctc_collate,
                               num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                               shuffle=False, collate_fn=ctc_collate,
                               num_workers=0)

    print(f"Train: {n_train}   Val: {n_val}\n")

    # Model
    model = DenseNetGRU_OCR(vocab_size=tok.vocab_size,
                              gru_hidden=256, gru_layers=2, dropout=0.3)
    model.to(device)
    print(f"Parameters: {model.count_params():,}")
    print(f"  (DenseNet121 frozen for phase 1)\n")

    ctc_loss_fn = nn.CTCLoss(blank=tok.PAD, zero_infinity=True)

    # ── Phase 1: freeze DenseNet, train only GRU+FC (10 epochs) ──────────
    model.freeze_densenet(freeze=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    phase1_epochs = min(10, args.epochs // 3)
    print(f"Phase 1: {phase1_epochs} epochs (DenseNet frozen, training GRU+FC)")

    best_cer = 1.0
    history  = []

    for epoch in range(1, phase1_epochs + 1):
        history = _run_epoch(epoch, model, train_loader, val_loader,
                              optimizer, ctc_loss_fn, tok, device,
                              history, args.out, best_cer, phase="P1")
        best_cer = min(best_cer, history[-1]["cer"])

    # ── Phase 2: unfreeze all, lower LR (remaining epochs) ───────────────
    remaining = args.epochs - phase1_epochs
    if remaining > 0:
        model.freeze_densenet(freeze=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6
        )
        print(f"\nPhase 2: {remaining} epochs (all layers unfrozen, lr=1e-4)")

        for epoch in range(phase1_epochs + 1, args.epochs + 1):
            history = _run_epoch(epoch, model, train_loader, val_loader,
                                  optimizer, ctc_loss_fn, tok, device,
                                  history, args.out, best_cer, phase="P2")
            ep_cer   = history[-1]["cer"]
            best_cer = min(best_cer, ep_cer)
            scheduler.step(ep_cer)

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best CER: {best_cer:.4f}")


def _run_epoch(epoch, model, train_loader, val_loader,
                optimizer, ctc_loss_fn, tok, device,
                history, out_dir, best_cer, phase=""):
    # Train
    model.train()
    total_loss = 0.0
    blank_fracs = []

    for imgs, labels_flat, label_lens in train_loader:
        imgs        = imgs.to(device)
        labels_flat = labels_flat.to(device)
        label_lens  = label_lens.to(device)

        log_probs     = model(imgs)
        T, B, _       = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = ctc_loss_fn(log_probs, labels_flat, input_lengths, label_lens)
        if torch.isnan(loss): continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        bf = (log_probs.argmax(2) == tok.PAD).float().mean().item()
        blank_fracs.append(bf)

    avg_loss  = total_loss / max(1, len(train_loader))
    avg_blank = sum(blank_fracs) / max(1, len(blank_fracs))

    # Validate
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for imgs, labels_flat, label_lens in val_loader:
            imgs      = imgs.to(device)
            log_probs = model(imgs)
            pred_ids  = ctc_greedy_decode(log_probs.cpu(), tok.PAD)
            preds.extend([tok.decode(ids) for ids in pred_ids])
            offset = 0
            for length in label_lens.tolist():
                ref_ids = labels_flat[offset:offset+length].tolist()
                refs.append(tok.decode(ref_ids))
                offset += length

    avg_cer  = sum(cer(p,r) for p,r in zip(preds,refs)) / max(1,len(refs))
    word_acc = sum(p==r     for p,r in zip(preds,refs)) / max(1,len(refs))

    blank_warn = "  ← BLANK COLLAPSE ⚠" if avg_blank > 0.85 else ""
    print(f"  [{phase}] Ep {epoch:3d}  loss={avg_loss:.4f}  "
          f"CER={avg_cer:.3f}  WA={word_acc:.3f}  blank={avg_blank:.2f}"
          f"{blank_warn}")

    # Sample predictions every 10 epochs
    if epoch % 10 == 0:
        print(f"\n  Samples:")
        for p, r in list(zip(preds, refs))[:4]:
            icon = "✓" if p==r else "✗"
            print(f"    {icon}  ref={r!r:20s}  pred={p!r}")
        print()

    history.append({"epoch": epoch, "phase": phase,
                     "loss": round(avg_loss,5), "cer": round(avg_cer,5),
                     "word_acc": round(word_acc,5)})

    if avg_cer < best_cer:
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "vocab_size": tok.vocab_size, "best_cer": avg_cer,
        }, os.path.join(out_dir, "best.pt"))
        print(f"  ✓  Saved best  CER={avg_cer:.4f}\n")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Inference class (plug into app.py)
# ──────────────────────────────────────────────────────────────────────────────

class BengaliDenseNetOCR:
    """
    Inference wrapper — drop-in replacement for BengaliTrOCR in app.py.

    Usage in app.py
    ───────────────
    from bn_densenet_ocr import BengaliDenseNetOCR
    _ocr = BengaliDenseNetOCR("checkpoints_dense/best.pt")
    text = _ocr.recognize(crop_bgr)
    texts = _ocr.recognize_batch(crops)
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok    = BnGraphemeTokenizer()

        ckpt  = torch.load(checkpoint_path, map_location=self.device)
        model = DenseNetGRU_OCR(vocab_size=self.tok.vocab_size)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        self.model = model
        self.blank = self.tok.PAD
        print(f"[BengaliDenseNetOCR] Loaded from {checkpoint_path}")

    def recognize(self, img_bgr: np.ndarray) -> str:
        if img_bgr is None or img_bgr.size == 0:
            return ""
        try:
            t  = preprocess_word(img_bgr, augment=False).unsqueeze(0).to(self.device)
            with torch.no_grad():
                lp = self.model(t)
            ids = ctc_greedy_decode(lp.cpu(), self.blank)[0]
            return self.tok.decode(ids)
        except Exception as e:
            return ""

    def recognize_batch(self, crops: list, batch_size: int = 16) -> list:
        results = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i+batch_size]
            valid = [c for c in batch if c is not None and c.size > 0]
            if not valid:
                results.extend([""] * len(batch))
                continue
            tensors = torch.stack([
                preprocess_word(c, augment=False) for c in valid
            ]).to(self.device)
            with torch.no_grad():
                lp = self.model(tensors)
            pred_ids = ctc_greedy_decode(lp.cpu(), self.blank)
            results.extend([self.tok.decode(ids) for ids in pred_ids])
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Prediction CLI
# ──────────────────────────────────────────────────────────────────────────────

def predict(args):
    ocr   = BengaliDenseNetOCR(args.model)
    exts  = {".png",".jpg",".jpeg",".bmp"}
    files = sorted([p for p in Path(args.words).iterdir()
                    if p.suffix.lower() in exts])
    print(f"\n{'File':<45} {'Predicted'}")
    print("─"*70)
    for f in files:
        img  = cv2.imread(str(f))
        text = ocr.recognize(img) if img is not None else ""
        print(f"{f.name:<45} {text!r}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",    required=True, choices=["train","predict"])
    p.add_argument("--labels",  default="labels.csv")
    p.add_argument("--images",  default="output/words/")
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--out",     default="checkpoints_dense")
    p.add_argument("--model",   default="checkpoints_dense/best.pt")
    p.add_argument("--words",   default="output/words/")
    args = p.parse_args()

    if args.mode == "train":
        train(args)
    else:
        predict(args)