"""
main.py
Entry point for the Image Captioning CLI.

Usage examples
──────────────
  Single image (BLIP — default):
    python main.py image.jpg

  Single image from URL:
    python main.py https://example.com/cat.jpg

  Batch — caption all images in a folder:
    python main.py --folder ./images

  Use ResNet feature-extraction mode (no HuggingFace):
    python main.py image.jpg --mode resnet

  Save results to a text file:
    python main.py --folder ./images --output captions.txt

  CPU-only, more beam width:
    python main.py image.jpg --device cpu --beams 6
"""

import argparse
import sys
import os
from pathlib import Path

# ── ANSI colour helpers ───────────────────────────────────────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    _CYAN   = "\033[96m"
    _GREEN  = "\033[92m"
    _YELLOW = "\033[93m"
    _RED    = "\033[91m"
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"
except ImportError:
    _CYAN = _GREEN = _YELLOW = _RED = _BOLD = _RESET = ""

DIVIDER = "─" * 58


def _banner() -> None:
    print(f"\n{_BOLD}{_CYAN}{'':>4}🖼️   IMAGE CAPTIONING AI   🖼️{_RESET}")
    print(DIVIDER)
    print("  ResNet-50 feature extractor  +  BLIP transformer decoder")
    print(DIVIDER + "\n")


def _ok(msg: str)   -> None: print(f"  {_GREEN}✔{_RESET}  {msg}")
def _info(msg: str) -> None: print(f"  {_CYAN}ℹ{_RESET}  {msg}")
def _warn(msg: str) -> None: print(f"  {_YELLOW}⚠{_RESET}  {msg}")
def _err(msg: str)  -> None: print(f"  {_RED}✘{_RESET}  {msg}", file=sys.stderr)


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Generate natural language captions for images using AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_grp = p.add_mutually_exclusive_group(required=True)
    input_grp.add_argument(
        "image",
        nargs="?",
        metavar="IMAGE",
        help="Path to a single image file, or a public image URL.",
    )
    input_grp.add_argument(
        "--folder", "-f",
        metavar="DIR",
        help="Caption all images in this directory.",
    )

    # Model options
    p.add_argument(
        "--mode", "-m",
        choices=["blip", "resnet"],
        default="blip",
        help="Captioning mode: 'blip' (transformer, default) or 'resnet' (template).",
    )
    p.add_argument(
        "--device", "-d",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device. 'auto' picks CUDA if available (default: auto).",
    )
    p.add_argument(
        "--beams", "-b",
        type=int,
        default=4,
        metavar="N",
        help="Beam search width for BLIP (higher = better, slower). Default: 4.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        metavar="N",
        help="Maximum caption length in tokens (BLIP only). Default: 50.",
    )

    # Output
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save captions to a text file (appends if exists).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress banner and progress; print only captions.",
    )

    return p


# ── Single-image captioning ───────────────────────────────────────────────────

def caption_single(args, captioner) -> str | None:
    from image_utils import load_image, image_info

    source = args.image
    try:
        _info(f"Loading image: {source}")
        img = load_image(source)
        info = image_info(img, source)
        _ok(f"Loaded  {info['width']} × {info['height']} px  ({info['mode']})")
    except (FileNotFoundError, ValueError) as exc:
        _err(str(exc))
        return None

    _info("Generating caption…")
    result = captioner.caption(img)

    print()
    print(f"  {_BOLD}Caption:{_RESET} {_GREEN}{result.caption}{_RESET}")
    if result.alternatives:
        print(f"  {_BOLD}Alt 1  :{_RESET} {result.alternatives[0]}")
    if result.alternatives and len(result.alternatives) > 1:
        print(f"  {_BOLD}Alt 2  :{_RESET} {result.alternatives[1]}")
    if result.top_labels:
        print(f"  {_BOLD}Labels :{_RESET} {', '.join(result.top_labels[:5])}")
    print(f"  {_BOLD}Time   :{_RESET} {result.inference_ms:.1f} ms")

    return result.caption


# ── Batch folder captioning ───────────────────────────────────────────────────

def caption_folder(args, captioner) -> list[tuple[str, str]]:
    from image_utils import load_image, list_images, image_info

    try:
        paths = list_images(args.folder)
    except NotADirectoryError as exc:
        _err(str(exc))
        return []

    if not paths:
        _warn(f"No supported images found in: {args.folder}")
        return []

    _info(f"Found {len(paths)} image(s) in '{args.folder}'")
    print()

    results: list[tuple[str, str]] = []

    for i, path in enumerate(paths, 1):
        print(f"  {_CYAN}[{i}/{len(paths)}]{_RESET} {path.name}")
        try:
            img = load_image(str(path))
            result = captioner.caption(img)
            print(f"    → {_GREEN}{result.caption}{_RESET}")
            print(f"    ⏱  {result.inference_ms:.0f} ms")
            results.append((str(path), result.caption))
        except Exception as exc:
            _warn(f"Skipped ({exc})")
        print()

    return results


# ── Save output ───────────────────────────────────────────────────────────────

def save_output(output_file: str, entries: list[tuple[str, str]]) -> None:
    mode = "a" if Path(output_file).exists() else "w"
    with open(output_file, mode, encoding="utf-8") as f:
        for source, caption in entries:
            f.write(f"{source}\t{caption}\n")
    _ok(f"Captions saved to: {output_file}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if not args.quiet:
        _banner()

    # ── Load model ──
    from captioner import ImageCaptioner
    cap = ImageCaptioner(
        mode=args.mode,
        device=args.device,
        num_beams=args.beams,
        max_new_tokens=args.max_tokens,
    )

    _info(f"Mode: {args.mode.upper()}  |  Device: {args.device}")
    cap.load_model(verbose=not args.quiet)

    print(DIVIDER)

    # ── Run ──
    entries: list[tuple[str, str]] = []

    if args.image:
        caption = caption_single(args, cap)
        if caption:
            entries.append((args.image, caption))
    else:
        entries = caption_folder(args, cap)

    # ── Summary ──
    if not args.quiet:
        print(DIVIDER)
        if entries:
            _ok(f"Done — captioned {len(entries)} image(s).")
        else:
            _warn("No images were successfully captioned.")

    # ── Save ──
    if args.output and entries:
        save_output(args.output, entries)

    print()


if __name__ == "__main__":
    main()
