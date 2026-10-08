"""Utility functions for image processing, validation, style conditioning, and file storage."""

from __future__ import annotations

import io
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from app.config import (
    STYLE_PRESETS,
    TEXT_TO_IMAGE_DIR,
    STYLE_TRANSFER_DIR,
    VARIATIONS_DIR,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


class ImageStudioError(Exception):
    """Base exception for AI Image Generation Studio."""
    pass


class InvalidImageError(ImageStudioError):
    """Raised when an uploaded or processed image is invalid."""
    pass


class InvalidPromptError(ImageStudioError):
    """Raised when a user prompt fails validation."""
    pass


class PipelineExecutionError(ImageStudioError):
    """Raised when model inference or style transfer fails."""
    pass


def validate_prompt(prompt: str, max_length: int = 1000) -> str:
    """Validate user prompt text."""
    if not prompt or not prompt.strip():
        raise InvalidPromptError("The prompt cannot be empty. Please enter a descriptive prompt.")
    clean = prompt.strip()
    if len(clean) > max_length:
        raise InvalidPromptError(f"Prompt exceeds maximum character length of {max_length}.")
    return clean


def apply_style_preset(
    base_prompt: str,
    preset_name: str,
    user_negative_prompt: str = "",
) -> Tuple[str, str]:
    """Condition the user prompt with aesthetic descriptors based on selected preset."""
    preset = STYLE_PRESETS.get(preset_name, STYLE_PRESETS["None"])
    suffix = preset.get("prompt_suffix", "")
    preset_neg = preset.get("negative_prompt", "")

    # Combine positive prompt
    final_prompt = f"{base_prompt.strip()}{suffix}" if suffix else base_prompt.strip()

    # Combine negative prompts
    neg_parts = []
    if user_negative_prompt and user_negative_prompt.strip():
        neg_parts.append(user_negative_prompt.strip())
    if preset_neg:
        neg_parts.append(preset_neg)
    final_negative_prompt = ", ".join(neg_parts)

    return final_prompt, final_negative_prompt


def validate_and_load_image(
    image_source: Union[bytes, io.BytesIO, Path, str, Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    max_dimension: Optional[int] = 1024,
) -> Image.Image:
    """Validate, format, and load an image into standard RGB PIL Image."""
    try:
        if isinstance(image_source, Image.Image):
            img = image_source.copy()
        else:
            if isinstance(image_source, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_source))
            elif isinstance(image_source, io.BytesIO):
                image_source.seek(0)
                img = Image.open(image_source)
            elif isinstance(image_source, (str, Path)):
                img = Image.open(image_source)
            else:
                raise InvalidImageError(f"Unsupported image input type: {type(image_source)}")

            # Ensure image integrity
            img.verify()

            # Re-open after verify()
            if isinstance(image_source, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_source))
            elif isinstance(image_source, io.BytesIO):
                image_source.seek(0)
                img = Image.open(image_source)
            elif isinstance(image_source, (str, Path)):
                img = Image.open(image_source)

        # Handle EXIF orientation
        img = ImageOps.exif_transpose(img)

        # Convert to standard RGB (drops alpha channel cleanly with white background if transparent)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")

        # Resize if specified
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        elif max_dimension is not None:
            w, h = img.size
            if max(w, h) > max_dimension:
                scale = max_dimension / float(max(w, h))
                new_w, new_h = int(w * scale), int(h * scale)
                # Keep dimensions divisible by 8 for diffusion latent spaces
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                img = img.resize((max(8, new_w), max(8, new_h)), Image.Resampling.LANCZOS)

        return img

    except Exception as e:
        if isinstance(e, InvalidImageError):
            raise
        raise InvalidImageError(f"Failed to process image: {str(e)}") from e


def save_image_with_metadata(
    image: Image.Image,
    output_dir: Path,
    prefix: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Save generated image and accompanying JSON metadata sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    base_name = f"{prefix}_{timestamp}_{unique_id}"

    image_path = output_dir / f"{base_name}.png"
    meta_path = output_dir / f"{base_name}.json"

    # Save PNG
    image.save(image_path, format="PNG", optimize=True)

    # Save metadata
    full_meta = {
        "id": unique_id,
        "filename": image_path.name,
        "timestamp": datetime.now().isoformat(),
        "width": image.width,
        "height": image.height,
        "prefix": prefix,
        **(metadata or {}),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)

    return image_path, meta_path


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Serialize PIL Image to byte buffer for web downloads."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def get_gallery_records(
    category: Optional[str] = None,
    limit: int = 48,
) -> List[Dict[str, Any]]:
    """Retrieve gallery items without loading all full-resolution images into memory."""
    directories: List[Tuple[Path, str]] = []
    if category == "text_to_image" or category is None:
        directories.append((TEXT_TO_IMAGE_DIR, "Text-to-Image"))
    if category == "style_transfer" or category is None:
        directories.append((STYLE_TRANSFER_DIR, "Style Transfer"))
    if category == "variations" or category is None:
        directories.append((VARIATIONS_DIR, "Variations"))

    records: List[Dict[str, Any]] = []

    for d_path, cat_name in directories:
        if not d_path.exists():
            continue
        # Scan for images
        for img_file in d_path.glob("*.png"):
            meta_file = img_file.with_suffix(".json")
            meta_data: Dict[str, Any] = {}
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                except Exception:
                    pass

            created_time = img_file.stat().st_mtime
            records.append({
                "path": str(img_file),
                "category": cat_name,
                "created_time": created_time,
                "metadata": meta_data,
                "prompt": meta_data.get("prompt", meta_data.get("style_name", "")),
                "parameters": meta_data,
            })

    # Sort descending by creation time (newest first)
    records.sort(key=lambda r: r["created_time"], reverse=True)
    return records[:limit]

