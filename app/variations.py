"""Image Variations pipeline generating 4 distinct variations using Diffusers Img2Img."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image

from app.config import VARIATIONS_DIR, config
from app.hardware import cleanup_memory, detect_hardware, get_torch_device
from app.utils import (
    PipelineExecutionError,
    save_image_with_metadata,
    validate_and_load_image,
)

logger = logging.getLogger(__name__)

# Cached pipeline instance
_CACHED_IMG2IMG_PIPE = None
_CACHED_PIPE_MODEL_ID: Optional[str] = None
_CACHED_PIPE_DEVICE: Optional[torch.device] = None


def load_img2img_pipeline(
    model_id: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_fp16: Optional[bool] = None,
) -> Any:
    """Lazily load and cache Hugging Face Image-to-Image pipeline."""
    global _CACHED_IMG2IMG_PIPE, _CACHED_PIPE_MODEL_ID, _CACHED_PIPE_DEVICE

    if model_id is None:
        model_id = config.model_id
    if device is None:
        device = get_torch_device(config.device_mode)
    if use_fp16 is None:
        use_fp16 = config.use_fp16 and (device.type == "cuda")

    if (
        _CACHED_IMG2IMG_PIPE is not None
        and _CACHED_PIPE_MODEL_ID == model_id
        and _CACHED_PIPE_DEVICE == device
    ):
        return _CACHED_IMG2IMG_PIPE

    cleanup_memory()

    try:
        from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
    except ImportError as exc:
        raise PipelineExecutionError(
            "Missing 'diffusers' package. Please install diffusers with `pip install diffusers`."
        ) from exc

    torch_dtype = torch.float16 if (device.type == "cuda" and use_fp16) else torch.float32

    logger.info(f"Loading Image Variations pipeline: {model_id} on {device} (dtype={torch_dtype})...")

    try:
        # Check if text-to-image pipeline is already loaded to share weights
        from app.text_to_image import _CACHED_TXT2IMG_PIPE, _CACHED_PIPE_MODEL_ID as txt_model, _CACHED_PIPE_DEVICE as txt_dev
        if _CACHED_TXT2IMG_PIPE is not None and txt_model == model_id and txt_dev == device:
            try:
                pipe = AutoPipelineForImage2Image.from_pipe(_CACHED_TXT2IMG_PIPE)
                _CACHED_IMG2IMG_PIPE = pipe
                _CACHED_PIPE_MODEL_ID = model_id
                _CACHED_PIPE_DEVICE = device
                return _CACHED_IMG2IMG_PIPE
            except Exception:
                pass

        pipeline_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
        }
        if config.hf_token:
            pipeline_kwargs["token"] = config.hf_token

        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **pipeline_kwargs)
        except Exception:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **pipeline_kwargs)

        if device.type == "cuda":
            pipe = pipe.to(device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()

            hw = detect_hardware()
            if hw.vram_total_mb > 0 and hw.vram_total_mb <= 4500:
                try:
                    pipe.enable_model_cpu_offload()
                except Exception as offload_err:
                    logger.debug(f"enable_model_cpu_offload notice: {offload_err}")
        else:
            pipe = pipe.to(device)

        _CACHED_IMG2IMG_PIPE = pipe
        _CACHED_PIPE_MODEL_ID = model_id
        _CACHED_PIPE_DEVICE = device

        return _CACHED_IMG2IMG_PIPE

    except Exception as e:
        cleanup_memory()
        logger.error(f"Failed to load Image Variations pipeline: {e}")
        raise PipelineExecutionError(
            f"Unable to load model '{model_id}' for variations: {str(e)}"
        ) from e


def generate_image_variations(
    input_image: Image.Image,
    num_variations: int = 4,
    strength: float = 0.75,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, distorted, bad anatomy",
    base_seed: Optional[int] = None,
    image_size: int = 512,
    device: Optional[torch.device] = None,
    model_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Tuple[Image.Image, Path, Dict[str, Any]]]:
    """Generate exactly `num_variations` (default: 4) using distinct seeds.

    Args:
        input_image: Source PIL image.
        num_variations: Number of variations to generate (default: 4).
        strength: Transformation strength from 0.0 to 1.0 (default: 0.75).
        num_inference_steps: Denoising steps (default: 25).
        guidance_scale: Classifier-free guidance weight (default: 7.5).
        prompt: Optional conditioning prompt.
        negative_prompt: Negative guidance prompt.
        base_seed: Starting seed for variations.
        image_size: Target square dimension (default: 512).
        device: Execution device.
        model_id: Diffusers model identifier.
        progress_callback: Callback reporting (current_variation_index, total_variations).

    Returns:
        List of tuples: (generated_image, saved_path, metadata_dict).
    """
    if device is None:
        device = get_torch_device(config.device_mode)

    # Validate and standardize input image to 512x512 RGB
    clean_image = validate_and_load_image(input_image, target_size=(image_size, image_size))

    # Fallback prompt for variations if none provided
    effective_prompt = prompt.strip() if prompt and prompt.strip() else "high quality masterpiece variation, intricate details"

    pipe = load_img2img_pipeline(model_id=model_id, device=device)

    # Determine unique seeds for each variation
    if base_seed is None or base_seed < 0:
        base_seed = random.randint(0, 2**30)

    seeds = [base_seed + (i * 1337) for i in range(num_variations)]
    results: List[Tuple[Image.Image, Path, Dict[str, Any]]] = []

    for idx, var_seed in enumerate(seeds):
        logger.info(f"Generating variation {idx + 1}/{num_variations} with seed {var_seed}...")

        generator = torch.Generator(device=device if device.type == "cpu" else "cuda").manual_seed(var_seed)

        inference_args: Dict[str, Any] = {
            "prompt": effective_prompt,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "image": clean_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        try:
            with torch.inference_mode():
                out = pipe(**inference_args)
            var_img = out.images[0]
        except torch.cuda.OutOfMemoryError as oom:
            cleanup_memory()
            logger.warning(f"CUDA OOM during variation {idx + 1}. Retrying variation on CPU...")
            pipe_cpu = load_img2img_pipeline(model_id=model_id, device=torch.device("cpu"), use_fp16=False)
            gen_cpu = torch.Generator(device="cpu").manual_seed(var_seed)
            inference_args["generator"] = gen_cpu
            with torch.inference_mode():
                out = pipe_cpu(**inference_args)
            var_img = out.images[0]
        except Exception as err:
            cleanup_memory()
            raise PipelineExecutionError(f"Failed generating variation {idx + 1}: {err}") from err

        meta: Dict[str, Any] = {
            "variation_index": idx + 1,
            "total_variations": num_variations,
            "seed": var_seed,
            "base_seed": base_seed,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "prompt": effective_prompt,
            "width": image_size,
            "height": image_size,
            "model_id": model_id or config.model_id,
        }

        img_path, _ = save_image_with_metadata(
            image=var_img,
            output_dir=VARIATIONS_DIR,
            prefix=f"variation_{idx + 1}",
            metadata=meta,
        )

        results.append((var_img, img_path, meta))

        if progress_callback:
            progress_callback(idx + 1, num_variations)

    cleanup_memory()
    return results

