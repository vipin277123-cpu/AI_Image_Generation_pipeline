"""Unit tests for image utilities, validation, and storage functions."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from app.utils import (
    InvalidImageError,
    InvalidPromptError,
    apply_style_preset,
    get_gallery_records,
    image_to_bytes,
    save_image_with_metadata,
    validate_and_load_image,
    validate_prompt,
)


class TestUtils(unittest.TestCase):
    """Test suite for image utilities and conditioning."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_prompt_valid(self):
        """Test valid prompts pass through cleanly."""
        prompt = "A majestic lion in savanna"
        self.assertEqual(validate_prompt(prompt), prompt)

    def test_validate_prompt_empty_raises(self):
        """Test empty prompt raises InvalidPromptError."""
        with self.assertRaises(InvalidPromptError):
            validate_prompt("")
        with self.assertRaises(InvalidPromptError):
            validate_prompt("   ")

    def test_validate_prompt_exceeds_max_length(self):
        """Test prompt exceeding max length raises InvalidPromptError."""
        long_prompt = "a" * 1500
        with self.assertRaises(InvalidPromptError):
            validate_prompt(long_prompt, max_length=100)

    def test_apply_style_preset_none(self):
        """Test style preset 'None' preserves original prompt."""
        p, neg = apply_style_preset("A mountain", "None")
        self.assertEqual(p, "A mountain")
        self.assertIn("blurry", neg)

    def test_apply_style_preset_cyberpunk(self):
        """Test style preset conditioning appends descriptors."""
        p, neg = apply_style_preset("A car", "Cyberpunk", user_negative_prompt="grainy")
        self.assertIn("cyberpunk", p.lower())
        self.assertIn("neon", p.lower())
        self.assertIn("grainy", neg)

    def test_validate_and_load_image_from_pil(self):
        """Test validating an in-memory PIL Image."""
        test_img = Image.new("RGB", (200, 200), color=(255, 0, 0))
        processed = validate_and_load_image(test_img, target_size=(512, 512))
        self.assertEqual(processed.size, (512, 512))
        self.assertEqual(processed.mode, "RGB")

    def test_validate_and_load_image_rgba_conversion(self):
        """Test RGBA transparent image is properly composited to RGB."""
        rgba_img = Image.new("RGBA", (100, 100), color=(0, 255, 0, 128))
        buf = io.BytesIO()
        rgba_img.save(buf, format="PNG")
        buf.seek(0)

        processed = validate_and_load_image(buf.getvalue())
        self.assertEqual(processed.mode, "RGB")

    def test_validate_and_load_image_corrupted_raises(self):
        """Test corrupted image bytes raise InvalidImageError."""
        corrupted = b"not_an_image_file_content"
        with self.assertRaises(InvalidImageError):
            validate_and_load_image(corrupted)

    def test_save_image_with_metadata(self):
        """Test saving image creates both PNG and JSON sidecar."""
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        meta = {"prompt": "test prompt", "seed": 42}
        img_p, meta_p = save_image_with_metadata(
            image=img,
            output_dir=self.temp_dir,
            prefix="test_prefix",
            metadata=meta,
        )

        self.assertTrue(img_p.exists())
        self.assertTrue(meta_p.exists())
        self.assertEqual(img_p.suffix, ".png")
        self.assertEqual(meta_p.suffix, ".json")

    def test_image_to_bytes(self):
        """Test converting image to valid byte buffer."""
        img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        b = image_to_bytes(img, format="PNG")
        self.assertIsInstance(b, bytes)
        self.assertGreater(len(b), 0)


if __name__ == "__main__":
    unittest.main()

