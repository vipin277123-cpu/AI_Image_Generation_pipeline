"""Hardware detection, GPU management, and memory diagnostics for AI Image Generation Studio."""

from __future__ import annotations

import gc
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareReport:
    """Detailed hardware capabilities report."""

    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float

    # GPU / Accelerator
    gpu_available: bool
    cuda_available: bool
    gpu_name: str
    vram_total_mb: float
    vram_allocated_mb: float
    vram_free_mb: float

    # System & Framework
    python_version: str
    pytorch_version: str
    active_device: str
    is_cpu_fallback: bool
    warning: Optional[str] = None

    def summary_lines(self) -> list[str]:
        """Return formatted bullet points for display."""
        lines = [
            f"**CPU:** {self.cpu_name} ({self.cpu_cores_logical} vCPUs, {self.total_ram_gb:.1f} GB RAM)",
            f"**GPU Available:** {'Yes' if self.gpu_available else 'No'}",
            f"**GPU:** {self.gpu_name}",
            f"**CUDA Support:** {'Available' if self.cuda_available else 'Unavailable (CPU Mode)'}",
        ]
        if self.gpu_available and self.vram_total_mb > 0:
            lines.append(f"**VRAM:** {self.vram_total_mb:.0f} MB (Free: {self.vram_free_mb:.0f} MB)")
        lines.append(f"**Active Device:** `{self.active_device}`")
        if self.warning:
            lines.append(f"⚠️ **Note:** {self.warning}")
        return lines


def _query_system_cpu() -> str:
    """Retrieve detailed CPU name across platforms."""
    name = platform.processor() or "Unknown CPU"
    # On Windows, platform.processor() often returns an arch string like AMD64 Family...
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val:
                return str(val).strip()
        except Exception:
            pass
    return name


def _query_total_ram_gb() -> float:
    """Retrieve total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _query_nvidia_smi() -> Optional[Dict[str, Any]]:
    """Query nvidia-smi if available to find physical GPU specifications."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                total_mb = float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else 0.0
                free_mb = float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else 0.0
                return {"name": name, "total_mb": total_mb, "free_mb": free_mb}
    except Exception:
        pass
    return None


def detect_hardware(preferred_device_mode: str = "auto") -> HardwareReport:
    """Detect system hardware and recommend optimal PyTorch execution device."""
    cpu_name = _query_system_cpu()
    cpu_logical = os.cpu_count() or 1
    cpu_physical = max(1, cpu_logical // 2)
    try:
        import psutil
        cpu_physical = psutil.cpu_count(logical=False) or cpu_physical
    except Exception:
        pass

    ram_gb = _query_total_ram_gb()
    cuda_available = torch.cuda.is_available()

    gpu_available = False
    gpu_name = "None"
    vram_total_mb = 0.0
    vram_allocated_mb = 0.0
    vram_free_mb = 0.0
    warning: Optional[str] = None

    if cuda_available:
        gpu_available = True
        try:
            device_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_idx)
            props = torch.cuda.get_device_properties(device_idx)
            vram_total_mb = props.total_memory / (1024 * 1024)
            vram_allocated_mb = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
            vram_reserved_mb = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
            vram_free_mb = max(0.0, vram_total_mb - vram_reserved_mb)
        except Exception as e:
            logger.warning(f"Failed to query torch.cuda properties: {e}")
            gpu_name = "NVIDIA CUDA Device"
    else:
        # Check if physical GPU exists on machine even if torch is currently CPU-only
        smi_info = _query_nvidia_smi()
        if smi_info:
            gpu_available = True
            gpu_name = smi_info["name"]
            vram_total_mb = smi_info["total_mb"]
            vram_free_mb = smi_info["free_mb"]
            warning = "Physical NVIDIA GPU detected, but PyTorch is running in CPU mode. Generation will use CPU fallback."
        else:
            gpu_name = "No dedicated NVIDIA GPU detected"
            warning = "Running in CPU-only mode. Generation will take longer than on a dedicated GPU."

    # Decide active device
    if preferred_device_mode == "cuda" and cuda_available:
        active_device = "cuda"
        is_cpu_fallback = False
    elif preferred_device_mode == "cpu":
        active_device = "cpu"
        is_cpu_fallback = True
        warning = "Manual CPU mode override is active."
    elif cuda_available:
        active_device = "cuda"
        is_cpu_fallback = False
    else:
        active_device = "cpu"
        is_cpu_fallback = True

    return HardwareReport(
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_physical,
        cpu_cores_logical=cpu_logical,
        total_ram_gb=ram_gb,
        gpu_available=gpu_available,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_total_mb=vram_total_mb,
        vram_allocated_mb=vram_allocated_mb,
        vram_free_mb=vram_free_mb,
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        active_device=active_device,
        is_cpu_fallback=is_cpu_fallback,
        warning=warning,
    )


def get_torch_device(mode: str = "auto") -> torch.device:
    """Return standard torch.device instance respecting hardware and settings."""
    if mode == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif mode == "cpu":
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_memory() -> None:
    """Run aggressive garbage collection and clear PyTorch accelerator cache."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.debug(f"CUDA cleanup note: {e}")

