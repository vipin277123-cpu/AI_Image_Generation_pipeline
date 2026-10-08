"""Neural Style Transfer implementation using VGG19, Gram matrices, and optimization."""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import VGG19_Weights, vgg19

from app.hardware import cleanup_memory, get_torch_device
from app.utils import PipelineExecutionError, validate_and_load_image

logger = logging.getLogger(__name__)

# ImageNet normalization statistics expected by VGG19
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalization(nn.Module):
    """Normalize input tensor with ImageNet mean and std."""

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float], device: torch.device):
        super().__init__()
        self.mean = torch.tensor(mean, device=device).view(-1, 1, 1)
        self.std = torch.tensor(std, device=device).view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    """Calculate Gram matrix for feature representation."""
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    g = torch.mm(features, features.t())
    # Normalize by total number of elements in feature map
    return g.div(b * c * h * w)


class ContentLoss(nn.Module):
    """Computes MSE content reconstruction loss."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(input_tensor, self.target)
        return input_tensor


class StyleLoss(nn.Module):
    """Computes MSE style reconstruction loss between Gram matrices."""

    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        g = gram_matrix(input_tensor)
        self.loss = F.mse_loss(g, self.target)
        return input_tensor


# Global cached VGG19 features model
_CACHED_VGG19: Optional[nn.Sequential] = None
_CACHED_DEVICE: Optional[torch.device] = None


def load_vgg19_features(device: torch.device) -> nn.Sequential:
    """Lazily load and cache VGG19 feature extractor in eval mode."""
    global _CACHED_VGG19, _CACHED_DEVICE
    if _CACHED_VGG19 is not None and _CACHED_DEVICE == device:
        return _CACHED_VGG19

    try:
        weights = VGG19_Weights.DEFAULT
        full_model = vgg19(weights=weights).features.to(device).eval()
        for param in full_model.parameters():
            param.requires_grad = False
        _CACHED_VGG19 = full_model
        _CACHED_DEVICE = device
        return _CACHED_VGG19
    except Exception as e:
        raise PipelineExecutionError(f"Failed to load VGG19 model weights: {e}") from e


def image_to_tensor(image: Image.Image, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Convert PIL image to preprocessed 4D torch tensor [1, 3, H, W]."""
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert optimized 4D tensor back into PIL Image."""
    clamped = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(clamped)


def build_style_transfer_model_and_losses(
    vgg_features: nn.Sequential,
    style_img_tensor: torch.Tensor,
    content_img_tensor: torch.Tensor,
    device: torch.device,
    content_layers: Optional[List[str]] = None,
    style_layers: Optional[List[str]] = None,
) -> Tuple[nn.Sequential, List[ContentLoss], List[StyleLoss]]:
    """Build sequential model containing normalization and loss evaluation hooks."""
    if content_layers is None:
        content_layers = ["conv_4"]
    if style_layers is None:
        style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    normalization = Normalization(IMAGENET_MEAN, IMAGENET_STD, device=device)
    content_losses: List[ContentLoss] = []
    style_losses: List[StyleLoss] = []

    model = nn.Sequential(normalization)

    i = 0  # increment every conv
    for layer in vgg_features.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            # In-place ReLU would corrupt loss evaluation
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            name = f"layer_{i}"

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img_tensor).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img_tensor).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model after the last loss layer to avoid redundant forward computation
    last_loss_idx = 0
    for idx in range(len(model) - 1, -1, -1):
        if isinstance(model[idx], (ContentLoss, StyleLoss)):
            last_loss_idx = idx
            break
    model = model[: last_loss_idx + 1]

    return model, content_losses, style_losses


def run_style_transfer(
    content_image: Image.Image,
    style_image: Image.Image,
    num_iterations: int = 150,
    content_weight: float = 1.0,
    style_weight: float = 1e5,
    image_size: int = 512,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Image.Image:
    """Execute optimization-based neural style transfer using VGG19.

    Args:
        content_image: Source content PIL image.
        style_image: Source style PIL image.
        num_iterations: Optimization steps (default: 150).
        content_weight: Alpha weight for content fidelity (default: 1.0).
        style_weight: Beta weight for artistic style fidelity (default: 1e5).
        image_size: Output resolution square dimension (e.g. 512 or 384).
        device: torch.device instance (auto-detected if None).
        progress_callback: Optional callable receiving (step, total_steps, current_loss).

    Returns:
        Final stylized PIL Image.
    """
    if device is None:
        device = get_torch_device("auto")

    # Clean previous memory allocations
    cleanup_memory()

    target_dims = (image_size, image_size)
    content_clean = validate_and_load_image(content_image, target_size=target_dims)
    style_clean = validate_and_load_image(style_image, target_size=target_dims)

    content_tensor = image_to_tensor(content_clean, target_dims, device)
    style_tensor = image_to_tensor(style_clean, target_dims, device)

    # Initialize input image with content image for fast, crisp convergence
    input_img = content_tensor.clone().requires_grad_(True)

    vgg_features = load_vgg19_features(device)
    model, content_losses, style_losses = build_style_transfer_model_and_losses(
        vgg_features=vgg_features,
        style_img_tensor=style_tensor,
        content_img_tensor=content_tensor,
        device=device,
    )

    # Use LBFGS optimizer for superior convergence speed and visual fidelity
    optimizer = torch.optim.LBFGS([input_img], lr=1.0)

    step = [0]

    def closure() -> float:
        # Clamp image tensor to [0, 1]
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        c_loss = sum(cl.loss for cl in content_losses) * content_weight
        s_loss = sum(sl.loss for sl in style_losses) * style_weight
        total_loss = c_loss + s_loss

        total_loss.backward()

        step[0] += 1
        current_loss_val = float(total_loss.item())

        if progress_callback and (step[0] % 10 == 0 or step[0] == num_iterations):
            progress_callback(min(step[0], num_iterations), num_iterations, current_loss_val)

        return current_loss_val

    try:
        while step[0] < num_iterations:
            optimizer.step(closure)
    except torch.cuda.OutOfMemoryError as oom:
        cleanup_memory()
        logger.warning(f"CUDA OOM in Style Transfer: {oom}. Retrying on CPU at lower resolution...")
        if device.type == "cuda":
            return run_style_transfer(
                content_image=content_image,
                style_image=style_image,
                num_iterations=min(num_iterations, 80),
                content_weight=content_weight,
                style_weight=style_weight,
                image_size=384,
                device=torch.device("cpu"),
                progress_callback=progress_callback,
            )
        raise PipelineExecutionError("Out of memory during Style Transfer.") from oom
    except Exception as e:
        raise PipelineExecutionError(f"Style transfer failed: {str(e)}") from e

    with torch.no_grad():
        input_img.clamp_(0, 1)

    result_image = tensor_to_image(input_img)
    cleanup_memory()
    return result_image

