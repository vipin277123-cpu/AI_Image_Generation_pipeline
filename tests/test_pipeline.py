"""Unit tests for pipeline components, style transfer math, and mock inference."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from PIL import Image

from app.config import config
from app.style_transfer import (
    ContentLoss,
    Normalization,
    StyleLoss,
    gram_matrix,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class TestPipelineComponents(unittest.TestCase):
    """Test suite for neural style transfer and pipeline operations."""

    def test_gram_matrix_shape_and_properties(self):
        """Test that Gram matrix is square (C x C) and symmetric."""
        batch_size, channels, height, width = 1, 16, 32, 32
        features = torch.randn(batch_size, channels, height, width)

        g = gram_matrix(features)
        # Shape should be channels x channels
        self.assertEqual(g.shape, (channels, channels))

        # Gram matrix must be symmetric: G == G^T
        self.assertTrue(torch.allclose(g, g.t(), atol=1e-5))

        # Diagonal elements must be non-negative
        diag = torch.diagonal(g)
        self.assertTrue(torch.all(diag >= 0))

    def test_content_loss_calculation(self):
        """Test ContentLoss forward computation."""
        target = torch.randn(1, 8, 16, 16)
        content_loss_module = ContentLoss(target)

        # Same input as target should yield 0 loss
        content_loss_module(target)
        self.assertAlmostEqual(float(content_loss_module.loss.item()), 0.0, places=5)

        # Perturbed input should produce positive loss
        perturbed = target + 0.5
        content_loss_module(perturbed)
        self.assertGreater(float(content_loss_module.loss.item()), 0.0)

    def test_style_loss_calculation(self):
        """Test StyleLoss forward computation."""
        style_feature = torch.randn(1, 8, 16, 16)
        style_loss_module = StyleLoss(style_feature)

        # Same feature representation should yield near 0 loss
        style_loss_module(style_feature)
        self.assertAlmostEqual(float(style_loss_module.loss.item()), 0.0, places=5)

    def test_normalization_layer(self):
        """Test Normalization layer applies correct ImageNet shift and scale."""
        norm = Normalization(IMAGENET_MEAN, IMAGENET_STD, device=torch.device("cpu"))
        ones_img = torch.ones(1, 3, 10, 10)
        normalized = norm(ones_img)

        # Expected channel 0 value: (1.0 - 0.485) / 0.229 = ~2.2489
        expected_ch0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        self.assertAlmostEqual(float(normalized[0, 0, 0, 0].item()), expected_ch0, places=3)

    @patch("app.text_to_image.load_text_to_image_pipeline")
    def test_text_to_image_mock_generation(self, mock_load):
        """Test text-to-image orchestration with mock pipeline."""
        import tempfile
        from pathlib import Path
        from app.text_to_image import generate_text_to_image

        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        mock_output.images = [mock_img]
        mock_pipe.return_value = mock_output
        mock_load.return_value = mock_pipe

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("app.text_to_image.TEXT_TO_IMAGE_DIR", Path(tmp_dir)):
                img, path, meta = generate_text_to_image(
                    prompt="A serene forest at dawn",
                    num_inference_steps=5,
                    guidance_scale=7.0,
                    seed=123,
                    device=torch.device("cpu"),
                )

                self.assertIsInstance(img, Image.Image)
                self.assertEqual(img.size, (512, 512))
                self.assertEqual(meta["seed"], 123)
                self.assertEqual(meta["num_inference_steps"], 5)
                self.assertTrue(path.exists())

    @patch("app.variations.load_img2img_pipeline")
    def test_variations_mock_generation(self, mock_load_img):
        """Test that variations generate exactly 4 variations with unique seeds."""
        import tempfile
        from pathlib import Path
        from app.variations import generate_image_variations

        mock_pipe = MagicMock()
        mock_img = Image.new("RGB", (512, 512), color=(50, 150, 250))
        mock_pipe.return_value.images = [mock_img]
        mock_load_img.return_value = mock_pipe

        input_img = Image.new("RGB", (256, 256), color=(20, 20, 20))
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("app.variations.VARIATIONS_DIR", Path(tmp_dir)):
                results = generate_image_variations(
                    input_image=input_img,
                    num_variations=4,
                    strength=0.7,
                    num_inference_steps=5,
                    base_seed=42,
                    device=torch.device("cpu"),
                )

                self.assertEqual(len(results), 4)
                seeds = [meta["seed"] for _, _, meta in results]
                # Verify 4 distinct seeds
                self.assertEqual(len(set(seeds)), 4)
                for _, path, meta in results:
                    self.assertTrue(path.exists())
                    self.assertEqual(meta["strength"], 0.7)


if __name__ == "__main__":
    unittest.main()

