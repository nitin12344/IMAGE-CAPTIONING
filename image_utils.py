"""
image_utils.py
Image loading, validation, and preprocessing utilities.

Provides:
  - load_image()          : open & validate an image from a file path or URL
  - preprocess_resnet()   : transform an image for ResNet / VGG feature extraction
  - preprocess_blip()     : prepare a PIL image for the BLIP processor
  - list_images()         : list all supported image files in a directory
  - image_info()          : return basic metadata about an image
"""

import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

# ── Supported extensions ────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

# ── ImageNet normalisation constants (used by ResNet / VGG) ─────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(source: str) -> Image.Image:
    """
    Load a PIL Image from a local file path or a public HTTP/HTTPS URL.

    Parameters
    ----------
    source : str
        Local path (absolute or relative) or URL starting with http/https.

    Returns
    -------
    PIL.Image.Image  in RGB mode.

    Raises
    ------
    FileNotFoundError   — local file does not exist.
    ValueError          — URL is unreachable or source is not a valid image.
    """
    source = source.strip()

    if _is_url(source):
        return _load_from_url(source)
    return _load_from_file(source)


def _is_url(source: str) -> bool:
    return re.match(r"^https?://", source, re.IGNORECASE) is not None


def _load_from_file(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")
    try:
        img = Image.open(p).convert("RGB")
        img.load()  # force decode to catch corrupt files early
        return img
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image file (may be corrupt): {path}")


def _load_from_url(url: str) -> Image.Image:
    try:
        headers = {"User-Agent": "ImageCaptioner/1.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            from io import BytesIO
            data = BytesIO(response.read())
            img = Image.open(data).convert("RGB")
            img.load()
            return img
    except urllib.error.URLError as exc:
        raise ValueError(f"Could not fetch image from URL: {url}\n  Reason: {exc.reason}")
    except UnidentifiedImageError:
        raise ValueError(f"URL does not point to a recognisable image: {url}")


# ---------------------------------------------------------------------------
# Preprocessing — ResNet / VGG feature extraction
# ---------------------------------------------------------------------------

def preprocess_resnet(image: Image.Image):
    """
    Apply standard ImageNet transforms for ResNet / VGG feature extraction.

    Returns a (1, 3, 224, 224) float32 torch.Tensor.
    """
    try:
        import torch
        from torchvision import transforms
    except ImportError:
        raise ImportError("torch and torchvision are required. Run: pip install torch torchvision")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image)          # (3, 224, 224)
    return tensor.unsqueeze(0)         # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Preprocessing — BLIP (handled by HuggingFace processor, but we keep PIL)
# ---------------------------------------------------------------------------

def preprocess_blip(image: Image.Image) -> Image.Image:
    """
    Ensure the image is in RGB mode and at a reasonable resolution for BLIP.
    The actual tensor conversion is done by BlipProcessor inside the model.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize very large images to prevent memory issues (BLIP handles up to ~1024px)
    max_side = 1024
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def list_images(directory: str) -> list[Path]:
    """
    Return a sorted list of Path objects for all supported image files
    in the given directory (non-recursive).
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    images = sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return images


def image_info(image: Image.Image, source: str = "") -> dict:
    """
    Return basic metadata about a PIL Image.

    Returns
    -------
    dict with keys: source, width, height, mode, format
    """
    return {
        "source": source,
        "width":  image.width,
        "height": image.height,
        "mode":   image.mode,
        "format": image.format or "unknown",
    }
