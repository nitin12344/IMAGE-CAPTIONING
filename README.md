# 🖼️ Image Captioning AI

Combines **computer vision** (ResNet-50 feature extraction) with **natural language processing** (BLIP transformer) to automatically generate human-readable captions for images.

---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │         IMAGE CAPTIONING             │
                         └─────────────────────────────────────┘

 Input Image
     │
     ▼
┌─────────────┐     ┌────────────────────────────────────────────┐
│  image_     │     │  BLIP Mode (default, recommended)          │
│  utils.py   │────▶│                                            │
│             │     │  ViT Visual Encoder                        │
│  • Load     │     │      │                                     │
│  • Validate │     │      ▼                                     │
│  • Resize   │     │  Cross-Attention (image ↔ text)            │
│  • Preproc  │     │      │                                     │
└─────────────┘     │      ▼                                     │
                    │  Language Decoder (transformer)            │
                    │      │                                     │
                    │      ▼                                     │
                    │  "a dog playing in the snow"               │
                    └────────────────────────────────────────────┘

                    ┌────────────────────────────────────────────┐
                    │  ResNet Mode (educational / lightweight)   │
                    │                                            │
                    │  ResNet-50 Backbone                        │
                    │      │                                     │
                    │      ▼                                     │
                    │  1000-dim ImageNet feature vector          │
                    │      │                                     │
                    │      ▼                                     │
                    │  Top-k class labels + confidence           │
                    │      │                                     │
                    │      ▼                                     │
                    │  Template: "A photo clearly showing a cat" │
                    └────────────────────────────────────────────┘
```

---

## Project Structure

```
IMAGE-CAPTIONING/
├── main.py              ← CLI entry point
├── captioner.py         ← Model loading & inference engine
├── image_utils.py       ← Image loading, preprocessing, helpers
├── test_captioning.py   ← 40+ unit tests (pytest)
├── requirements.txt     ← Dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads ~900 MB of model weights (cached automatically afterwards).

### 2. Caption a single image

```bash
python main.py photo.jpg
```

### 3. Caption from a URL

```bash
python main.py https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg
```

### 4. Batch — caption all images in a folder

```bash
python main.py --folder ./my_images
```

### 5. Use ResNet mode (no HuggingFace, lightweight)

```bash
python main.py photo.jpg --mode resnet
```

### 6. Save captions to a file

```bash
python main.py --folder ./images --output captions.txt
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `IMAGE` | — | Path to an image file or HTTP/HTTPS URL |
| `--folder DIR` | — | Caption all images in a directory |
| `--mode blip\|resnet` | `blip` | Captioning backend |
| `--device auto\|cpu\|cuda` | `auto` | Torch device (auto picks CUDA if available) |
| `--beams N` | `4` | Beam search width (BLIP only) |
| `--max-tokens N` | `50` | Max caption tokens (BLIP only) |
| `--output FILE` | — | Save `path\tcaption` lines to a file |
| `--quiet` | off | Suppress banner; print captions only |

---

## Models

### BLIP (default)
- **Full name**: Bootstrapped Language-Image Pre-training
- **Source**: Salesforce Research / HuggingFace
- **Model ID**: `Salesforce/blip-image-captioning-base`
- **Architecture**: ViT-B/16 visual encoder + BERT-style language decoder
- **Download size**: ~900 MB (cached in `~/.cache/huggingface/`)
- **Output**: Free-form English sentences ("a woman riding a bicycle on a sunny day")

### ResNet-50 (fallback / educational)
- **Source**: torchvision, pre-trained on ImageNet-1K
- **Architecture**: Deep residual network, 50 layers
- **Download size**: ~100 MB
- **Output**: Template sentence from top ImageNet class predictions
- **Use case**: Demonstrates the feature-extraction stage; no HuggingFace required

---

## Example Output

```
🖼️   IMAGE CAPTIONING AI   🖼️
──────────────────────────────────────────────────────────
  ResNet-50 feature extractor  +  BLIP transformer decoder
──────────────────────────────────────────────────────────

  ℹ  Mode: BLIP  |  Device: cpu
  Loading BLIP model → Salesforce/blip-image-captioning-base
  ℹ  Loading image: dog.jpg
  ✔  Loaded  1280 × 960 px  (RGB)
  ℹ  Generating caption…

  Caption : a golden retriever dog sitting on a green lawn
  Alt 1   : a dog sitting on the grass looking at the camera
  Alt 2   : a golden retriever sitting on a lawn outdoors
  Time    : 1842.3 ms
──────────────────────────────────────────────────────────
  ✔  Done — captioned 1 image(s).
```

---

## Running Tests

```bash
pytest test_captioning.py -v
```

Tests cover image loading, preprocessing, template captions, ResNet inference, CLI parsing, and file output. ResNet integration tests are auto-skipped if `torch` is not installed.

---

## Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| CNN feature extraction | `captioner.py` → `_load_resnet`, `_caption_resnet` |
| Transfer learning | ResNet-50 + BLIP both use ImageNet/web pre-training |
| Transformer encoder-decoder | BLIP's ViT encoder + language decoder |
| Beam search decoding | `num_beams` parameter in `_caption_blip` |
| ImageNet normalisation | `image_utils.py` → `preprocess_resnet` |
| Softmax + top-k | `_caption_resnet` → confidence scores |
| Batch inference | `caption_batch()` in `ImageCaptioner` |
