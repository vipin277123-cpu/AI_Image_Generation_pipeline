"""Unit tests for hardware detection and memory management."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from app.hardware import (
    HardwareReport,
    cleanup_memory,
    detect_hardware,
    get_torch_device,
)


class TestHardwareDetection(unittest.TestCase):
    """Test suite for hardware detection routines."""

    def test_detect_hardware_returns_report(self):
        """Test that detect_hardware produces a complete HardwareReport object."""
        report = detect_hardware(preferred_device_mode="auto")
        self.assertIsInstance(report, HardwareReport)
        self.assertTrue(len(report.cpu_name) > 0)
        self.assertGreater(report.cpu_cores_logical, 0)
        self.assertGreater(report.cpu_cores_physical, 0)
        self.assertGreaterEqual(report.total_ram_gb, 0.0)
        self.assertIn(report.active_device, ["cuda", "cpu"])
        self.assertTrue(isinstance(report.gpu_available, bool))
        self.assertTrue(isinstance(report.cuda_available, bool))

    def test_detect_hardware_summary_lines(self):
        """Test summary lines formatting."""
        report = detect_hardware()
        lines = report.summary_lines()
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 3)
        joined = "\n".join(lines)
        self.assertIn("CPU", joined)
        self.assertIn("GPU", joined)

    def test_get_torch_device_cpu_override(self):
        """Test explicit CPU device selection."""
        dev = get_torch_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_get_torch_device_cuda(self):
        """Test CUDA device selection or fallback."""
        if torch.cuda.is_available():
            dev = get_torch_device("cuda")
            self.assertEqual(dev.type, "cuda")
        else:
            dev = get_torch_device("cuda")
            self.assertEqual(dev.type, "cpu")

    def test_cleanup_memory_runs_safely(self):
        """Test that cleanup_memory runs without raising exceptions."""
        try:
            cleanup_memory()
        except Exception as e:
            self.fail(f"cleanup_memory raised an exception: {e}")

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_fallback_mode_when_cuda_disabled(self, mock_cuda):
        """Test report when CUDA is mocked as unavailable."""
        report = detect_hardware()
        self.assertFalse(report.cuda_available)
        self.assertEqual(report.active_device, "cpu")
        self.assertTrue(report.is_cpu_fallback)


if __name__ == "__main__":
    unittest.main()

