"""
test_captioning.py
Unit tests for the image captioning project.

Tests cover:
  - image_utils  : load, validate, preprocess, list_images
  - captioner    : CaptionResult, template caption builder, ResNet mode
  - main / CLI   : argument parser, output helpers

Run with:
    pytest test_captioning.py -v
"""

import os
import sys
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Helpers: create a simple in-memory test image
# ---------------------------------------------------------------------------

def _make_rgb_image(width: int = 64, height: int = 64):
    """Return a small solid-colour PIL Image for testing (no file I/O needed)."""
    from PIL import Image as PILImage
    import random
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return PILImage.new("RGB", (width, height), color=color)


def _make_rgba_image():
    from PIL import Image as PILImage
    return PILImage.new("RGBA", (32, 32), color=(100, 150, 200, 255))


# ── image_utils tests ────────────────────────────────────────────────────────

class TestImageUtils:

    # ── load_image ──

    def test_load_from_missing_file_raises(self):
        from image_utils import load_image
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_image_xyz.jpg")

    def test_load_from_non_image_file_raises(self, tmp_path):
        from image_utils import load_image
        bad = tmp_path / "text.txt"
        bad.write_text("not an image")
        with pytest.raises(ValueError):
            load_image(str(bad))

    def test_load_valid_image_from_file(self, tmp_path):
        from image_utils import load_image
        from PIL import Image as PILImage
        img = _make_rgb_image()
        path = tmp_path / "test.png"
        img.save(str(path))
        loaded = load_image(str(path))
        assert loaded.mode == "RGB"
        assert loaded.width == 64

    def test_load_rgba_converts_to_rgb(self, tmp_path):
        from image_utils import load_image
        img = _make_rgba_image()
        path = tmp_path / "rgba.png"
        img.save(str(path))
        loaded = load_image(str(path))
        assert loaded.mode == "RGB"

    # ── preprocess_resnet ──

    def test_preprocess_resnet_output_shape(self):
        try:
            import torch
            from image_utils import preprocess_resnet
        except ImportError:
            pytest.skip("torch not installed")
        img = _make_rgb_image(128, 128)
        tensor = preprocess_resnet(img)
        assert tensor.shape == (1, 3, 224, 224)
        assert str(tensor.dtype) == "torch.float32"

    # ── preprocess_blip ──

    def test_preprocess_blip_keeps_rgb(self):
        from image_utils import preprocess_blip
        img = _make_rgb_image()
        out = preprocess_blip(img)
        assert out.mode == "RGB"

    def test_preprocess_blip_downsizes_large_image(self):
        from image_utils import preprocess_blip
        from PIL import Image as PILImage
        large = PILImage.new("RGB", (2048, 1500))
        out = preprocess_blip(large)
        assert max(out.size) <= 1024

    def test_preprocess_blip_small_image_unchanged_size(self):
        from image_utils import preprocess_blip
        img = _make_rgb_image(200, 200)
        out = preprocess_blip(img)
        assert out.size == (200, 200)

    # ── list_images ──

    def test_list_images_returns_sorted(self, tmp_path):
        from image_utils import list_images
        from PIL import Image as PILImage
        for name in ["b.jpg", "a.png", "c.jpeg"]:
            PILImage.new("RGB", (10, 10)).save(str(tmp_path / name))
        (tmp_path / "notes.txt").write_text("skip me")
        imgs = list_images(str(tmp_path))
        names = [p.name for p in imgs]
        assert "notes.txt" not in names
        assert names == sorted(names)
        assert len(imgs) == 3

    def test_list_images_bad_dir_raises(self):
        from image_utils import list_images
        with pytest.raises(NotADirectoryError):
            list_images("/nonexistent/path/xyz")

    def test_list_images_empty_dir_returns_empty(self, tmp_path):
        from image_utils import list_images
        assert list_images(str(tmp_path)) == []

    # ── image_info ──

    def test_image_info_fields(self):
        from image_utils import image_info
        img = _make_rgb_image(100, 80)
        info = image_info(img, source="test.jpg")
        assert info["width"]  == 100
        assert info["height"] == 80
        assert info["mode"]   == "RGB"
        assert info["source"] == "test.jpg"

    # ── URL detection (internal helper) ──

    def test_is_url_detects_http(self):
        from image_utils import _is_url
        assert _is_url("http://example.com/img.jpg")
        assert _is_url("https://example.com/img.jpg")
        assert not _is_url("/local/path/img.jpg")
        assert not _is_url("relative/path.png")


# ── captioner tests ───────────────────────────────────────────────────────────

class TestCaptionResult:

    def test_str_contains_caption(self):
        from captioner import CaptionResult
        r = CaptionResult(
            caption="a dog sitting on grass",
            mode="blip",
            model_name="Salesforce/blip-image-captioning-base",
            inference_ms=123.4,
            alternatives=["a dog in a field"],
        )
        s = str(r)
        assert "a dog sitting on grass" in s
        assert "123.4" in s
        assert "a dog in a field" in s

    def test_str_resnet_shows_labels(self):
        from captioner import CaptionResult
        r = CaptionResult(
            caption="A photo likely showing a cat.",
            mode="resnet",
            model_name="ResNet-50",
            inference_ms=55.0,
            top_labels=["cat (82.0%)", "kitten (5.0%)"],
        )
        s = str(r)
        assert "cat" in s


class TestTemplateCaptionBuilder:

    def test_high_confidence_single(self):
        from captioner import _build_template_caption
        cap = _build_template_caption(["golden retriever"], [0.90])
        assert "golden retriever" in cap.lower()
        assert "clearly" in cap.lower()

    def test_medium_confidence(self):
        from captioner import _build_template_caption
        cap = _build_template_caption(["volcano", "mountain"], [0.55, 0.20])
        assert "likely" in cap.lower()
        assert "volcano" in cap.lower()

    def test_low_confidence(self):
        from captioner import _build_template_caption
        cap = _build_template_caption(["cat", "dog"], [0.20, 0.15])
        assert "possibly" in cap.lower()

    def test_two_labels_combined(self):
        from captioner import _build_template_caption
        cap = _build_template_caption(["pizza", "plate"], [0.65, 0.25])
        assert "pizza" in cap.lower()
        assert "plate" in cap.lower()

    def test_single_low_secondary_score(self):
        from captioner import _build_template_caption
        # secondary score < 0.10 → only primary in caption
        cap = _build_template_caption(["cat", "dog"], [0.80, 0.05])
        assert "cat" in cap.lower()
        assert "dog" not in cap.lower()

    def test_returns_string(self):
        from captioner import _build_template_caption
        result = _build_template_caption(["bird"], [0.60])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_underscore_labels_cleaned(self):
        from captioner import _build_template_caption
        cap = _build_template_caption(["golden_retriever"], [0.80])
        assert "_" not in cap


class TestImageCaptionerInit:

    def test_valid_mode_blip(self):
        from captioner import ImageCaptioner
        cap = ImageCaptioner(mode="blip")
        assert cap.mode == "blip"

    def test_valid_mode_resnet(self):
        from captioner import ImageCaptioner
        cap = ImageCaptioner(mode="resnet")
        assert cap.mode == "resnet"

    def test_invalid_mode_raises(self):
        from captioner import ImageCaptioner
        with pytest.raises(ValueError, match="mode must be"):
            ImageCaptioner(mode="gpt4")

    def test_model_name_blip(self):
        from captioner import ImageCaptioner
        cap = ImageCaptioner(mode="blip")
        assert "blip" in cap.model_name.lower()

    def test_model_name_resnet(self):
        from captioner import ImageCaptioner
        cap = ImageCaptioner(mode="resnet")
        assert "resnet" in cap.model_name.lower()

    def test_device_cpu_set(self):
        from captioner import ImageCaptioner
        cap = ImageCaptioner(mode="resnet", device="cpu")
        assert cap.device == "cpu"

    def test_not_loaded_initially(self):
        from captioner import ImageCaptioner
        cap = ImageCaptioner(mode="resnet")
        assert cap._loaded is False


class TestCreateCaptioner:

    def test_factory_blip(self):
        from captioner import create_captioner
        cap = create_captioner(use_blip=True, device="cpu")
        assert cap.mode == "blip"

    def test_factory_resnet(self):
        from captioner import create_captioner
        cap = create_captioner(use_blip=False, device="cpu")
        assert cap.mode == "resnet"


class TestResNetCaptioner:
    """Integration-style tests for ResNet mode (requires torch + torchvision)."""

    @pytest.fixture(scope="class")
    def captioner(self):
        try:
            from captioner import ImageCaptioner
            cap = ImageCaptioner(mode="resnet", device="cpu")
            cap.load_model(verbose=False)
            return cap
        except ImportError:
            pytest.skip("torch/torchvision not installed")

    def test_caption_returns_result(self, captioner):
        from captioner import CaptionResult
        img = _make_rgb_image(224, 224)
        result = captioner.caption(img)
        assert isinstance(result, CaptionResult)

    def test_caption_is_non_empty_string(self, captioner):
        img = _make_rgb_image()
        result = captioner.caption(img)
        assert isinstance(result.caption, str)
        assert len(result.caption) > 5

    def test_caption_has_top_labels(self, captioner):
        img = _make_rgb_image()
        result = captioner.caption(img)
        assert len(result.top_labels) > 0

    def test_inference_time_positive(self, captioner):
        img = _make_rgb_image()
        result = captioner.caption(img)
        assert result.inference_ms > 0

    def test_load_model_idempotent(self, captioner):
        # Calling load_model again should not raise or reset state
        captioner.load_model(verbose=False)
        assert captioner._loaded is True

    def test_batch_caption(self, captioner):
        images = [_make_rgb_image() for _ in range(3)]
        results = captioner.caption_batch(images, verbose=False)
        assert len(results) == 3
        for r in results:
            assert isinstance(r.caption, str)

    def test_mode_field_is_resnet(self, captioner):
        img = _make_rgb_image()
        result = captioner.caption(img)
        assert result.mode == "resnet"


# ── CLI / main tests ──────────────────────────────────────────────────────────

class TestCLIParser:

    def test_image_arg_parsed(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg"])
        assert args.image == "photo.jpg"

    def test_folder_arg_parsed(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["--folder", "./images"])
        assert args.folder == "./images"

    def test_mode_default_is_blip(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg"])
        assert args.mode == "blip"

    def test_mode_resnet_explicit(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg", "--mode", "resnet"])
        assert args.mode == "resnet"

    def test_device_default_is_auto(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg"])
        assert args.device == "auto"

    def test_beams_default(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg"])
        assert args.beams == 4

    def test_custom_beams(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg", "--beams", "6"])
        assert args.beams == 6

    def test_output_flag(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg", "--output", "out.txt"])
        assert args.output == "out.txt"

    def test_quiet_flag(self):
        from main import build_parser
        p = build_parser()
        args = p.parse_args(["photo.jpg", "--quiet"])
        assert args.quiet is True

    def test_image_and_folder_mutually_exclusive(self):
        from main import build_parser
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["photo.jpg", "--folder", "./images"])

    def test_no_input_exits(self):
        from main import build_parser
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])


class TestSaveOutput:

    def test_creates_new_file(self, tmp_path):
        from main import save_output
        out = str(tmp_path / "captions.txt")
        save_output(out, [("img.jpg", "a cat on a mat")])
        lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert "a cat on a mat" in lines[0]

    def test_tab_separated(self, tmp_path):
        from main import save_output
        out = str(tmp_path / "captions.txt")
        save_output(out, [("foo.jpg", "some caption")])
        content = Path(out).read_text(encoding="utf-8")
        assert "\t" in content

    def test_appends_to_existing(self, tmp_path):
        from main import save_output
        out = str(tmp_path / "captions.txt")
        save_output(out, [("a.jpg", "caption one")])
        save_output(out, [("b.jpg", "caption two")])
        lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_multiple_entries(self, tmp_path):
        from main import save_output
        out = str(tmp_path / "captions.txt")
        entries = [("img1.jpg", "cap1"), ("img2.jpg", "cap2"), ("img3.jpg", "cap3")]
        save_output(out, entries)
        lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
