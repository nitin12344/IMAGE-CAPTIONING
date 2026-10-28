"""
captioner.py
Core image captioning engine.

Two modes are supported:

  1. BLIP (default, recommended)
     ─────────────────────────────
     Uses Salesforce's BLIP (Bootstrapped Language-Image Pre-training) model
     via HuggingFace Transformers. BLIP is a transformer-based encoder-decoder:

       Image → ViT visual encoder → cross-attention → language decoder → caption

     Model: "Salesforce/blip-image-captioning-base"
     Requires: transformers, torch, Pillow

  2. ResNet + Template (fallback / educational)
     ────────────────────────────────────────────
     Uses a pre-trained ResNet-50 to extract a 1000-dim ImageNet class
     probability vector, then maps the top predictions to a human-readable
     template caption. This demonstrates the feature-extraction stage of a
     full image captioning pipeline.

     Requires: torch, torchvision, Pillow

Usage
─────
    from captioner import ImageCaptioner

    cap = ImageCaptioner(mode="blip")        # or mode="resnet"
    cap.load_model()

    from image_utils import load_image
    img = load_image("photo.jpg")
    result = cap.caption(img)
    print(result["caption"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class CaptionResult:
    """Holds the output of a single captioning run."""
    caption:        str
    mode:           str                    # "blip" or "resnet"
    model_name:     str
    inference_ms:   float                  # wall-clock inference time in ms
    top_labels:     list[str] = field(default_factory=list)   # ResNet mode only
    alternatives:   list[str] = field(default_factory=list)   # BLIP beam results

    def __str__(self) -> str:
        lines = [
            f"Caption : {self.caption}",
            f"Model   : {self.model_name}  [{self.mode}]",
            f"Time    : {self.inference_ms:.1f} ms",
        ]
        if self.alternatives:
            lines.append("Alt     : " + " | ".join(self.alternatives[:3]))
        if self.top_labels:
            lines.append("Labels  : " + ", ".join(self.top_labels[:5]))
        return "\n".join(lines)


# ── Base class ───────────────────────────────────────────────────────────────

class ImageCaptioner:
    """
    Unified image captioning interface.

    Parameters
    ----------
    mode : "blip" | "resnet"
        Which backend to use. Default is "blip".
    device : "cpu" | "cuda" | "auto"
        Torch device. "auto" picks CUDA if available, else CPU.
    num_beams : int
        Beam search width for BLIP (higher = better quality, slower).
    max_new_tokens : int
        Max caption length in tokens (BLIP only).
    """

    def __init__(
        self,
        mode: str = "blip",
        device: str = "auto",
        num_beams: int = 4,
        max_new_tokens: int = 50,
    ) -> None:
        if mode not in ("blip", "resnet"):
            raise ValueError(f"mode must be 'blip' or 'resnet', got '{mode}'")
        self.mode           = mode
        self.num_beams      = num_beams
        self.max_new_tokens = max_new_tokens
        self._model         = None
        self._processor     = None
        self._labels        = None      # ImageNet labels for ResNet mode
        self._loaded        = False

        # Resolve device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        if self.mode == "blip":
            return "Salesforce/blip-image-captioning-base"
        return "ResNet-50 (torchvision, ImageNet)"

    def load_model(self, verbose: bool = True) -> None:
        """Download/load the model weights. Safe to call multiple times."""
        if self._loaded:
            return
        if self.mode == "blip":
            self._load_blip(verbose)
        else:
            self._load_resnet(verbose)
        self._loaded = True

    def caption(self, image: Image.Image) -> CaptionResult:
        """
        Generate a caption for a single PIL Image.

        Calls load_model() automatically if not already loaded.
        """
        if not self._loaded:
            self.load_model(verbose=False)

        if self.mode == "blip":
            return self._caption_blip(image)
        return self._caption_resnet(image)

    def caption_batch(
        self,
        images: list[Image.Image],
        verbose: bool = True,
    ) -> list[CaptionResult]:
        """Caption a list of images, returning one CaptionResult per image."""
        if not self._loaded:
            self.load_model(verbose=verbose)

        results = []
        total = len(images)
        for i, img in enumerate(images, 1):
            if verbose:
                print(f"  [{i}/{total}] Captioning image {i}...", end="\r")
            results.append(self.caption(img))
        if verbose:
            print()   # newline after \r
        return results

    # ── BLIP backend ─────────────────────────────────────────────────────

    def _load_blip(self, verbose: bool) -> None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
        except ImportError:
            raise ImportError(
                "BLIP requires: pip install transformers torch torchvision Pillow"
            )

        if verbose:
            print(f"  Loading BLIP model → {self.model_name}")
            print(f"  Device: {self.device}  |  Beams: {self.num_beams}")
            print("  (First run downloads ~900 MB — cached afterwards)\n")

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = (
            BlipForConditionalGeneration
            .from_pretrained(self.model_name)
            .to(self.device)
        )
        self._model.eval()

    def _caption_blip(self, image: Image.Image) -> CaptionResult:
        import torch
        from image_utils import preprocess_blip

        img = preprocess_blip(image)
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            # Beam search — return multiple sequences for alternatives
            output_ids = self._model.generate(
                **inputs,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=min(self.num_beams, 3),
                early_stopping=True,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        captions = [
            self._processor.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]

        return CaptionResult(
            caption=captions[0],
            mode="blip",
            model_name=self.model_name,
            inference_ms=elapsed_ms,
            alternatives=captions[1:],
        )

    # ── ResNet backend (feature extraction + template) ───────────────────

    def _load_resnet(self, verbose: bool) -> None:
        try:
            import torch
            import torchvision.models as models
        except ImportError:
            raise ImportError(
                "ResNet mode requires: pip install torch torchvision"
            )

        if verbose:
            print("  Loading ResNet-50 (ImageNet pre-trained)...")
            print(f"  Device: {self.device}\n")

        import torchvision.models as models
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self._model = models.resnet50(weights=weights).to(self.device)
        self._model.eval()
        self._labels = weights.meta["categories"]   # 1000 ImageNet class names

    def _caption_resnet(self, image: Image.Image) -> CaptionResult:
        import torch
        import torch.nn.functional as F
        from image_utils import preprocess_resnet

        tensor = preprocess_resnet(image).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self._model(tensor)              # (1, 1000)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        probs = F.softmax(logits[0], dim=0)
        top_k = 5
        top_probs, top_idxs = torch.topk(probs, top_k)

        top_labels = [
            self._labels[idx.item()] for idx in top_idxs
        ]
        top_scores = [p.item() for p in top_probs]

        caption = _build_template_caption(top_labels, top_scores)

        return CaptionResult(
            caption=caption,
            mode="resnet",
            model_name="ResNet-50 (torchvision, ImageNet)",
            inference_ms=elapsed_ms,
            top_labels=[
                f"{lbl} ({sc:.1%})"
                for lbl, sc in zip(top_labels, top_scores)
            ],
        )


# ── Template caption builder (ResNet fallback) ───────────────────────────────

def _build_template_caption(labels: list[str], scores: list[float]) -> str:
    """
    Convert top ImageNet predictions into a simple English sentence.

    Examples
    --------
    ["golden retriever", "Labrador retriever"] → "A photo showing a golden retriever."
    ["pizza", "plate"]                          → "A photo showing a pizza on a plate."
    ["volcano", "mountain"]                     → "A photo showing a volcano in a mountainous scene."
    """
    # Clean labels: remove underscores, lowercase
    clean = [l.replace("_", " ").lower() for l in labels]

    primary   = clean[0]
    secondary = clean[1] if len(clean) > 1 else None
    conf      = scores[0]

    # Confidence qualifier
    if conf >= 0.70:
        certainty = "clearly showing"
    elif conf >= 0.40:
        certainty = "likely showing"
    else:
        certainty = "possibly showing"

    if secondary and scores[1] > 0.10:
        # Use both top predictions
        caption = f"A photo {certainty} a {primary}, with a {secondary} also visible."
    else:
        caption = f"A photo {certainty} a {primary}."

    return caption.capitalize()


# ── Convenience factory ───────────────────────────────────────────────────────

def create_captioner(
    use_blip: bool = True,
    device: str = "auto",
    **kwargs,
) -> ImageCaptioner:
    """
    Factory function.

    Parameters
    ----------
    use_blip : bool
        True  → BLIP transformer model (better captions).
        False → ResNet feature extraction + template (lighter, educational).
    """
    mode = "blip" if use_blip else "resnet"
    return ImageCaptioner(mode=mode, device=device, **kwargs)
