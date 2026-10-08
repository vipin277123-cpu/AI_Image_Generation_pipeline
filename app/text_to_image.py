"""Text-to-Image pipeline implementation with Hugging Face Diffusers, CUDA optimizations, and lazy caching."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from PIL import Image

from app.config import TEXT_TO_IMAGE_DIR, config
from app.hardware import cleanup_memory, detect_hardware, get_torch_device
from app.utils import (
    PipelineExecutionError,
    apply_style_preset,
    save_image_with_metadata,
    validate_prompt,
)

logger = logging.getLogger(__name__)

# Global cached pipeline reference
_CACHED_TXT2IMG_PIPE = None
_CACHED_PIPE_MODEL_ID: Optional[str] = None
_CACHED_PIPE_DEVICE: Optional[torch.device] = None


def load_text_to_image_pipeline(
    model_id: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_fp16: Optional[bool] = None,
) -> Any:
    """Lazily load and cache Hugging Face Stable Diffusion pipeline."""
    global _CACHED_TXT2IMG_PIPE, _CACHED_PIPE_MODEL_ID, _CACHED_PIPE_DEVICE

    if model_id is None:
        model_id = config.model_id
    if device is None:
        device = get_torch_device(config.device_mode)
    if use_fp16 is None:
        use_fp16 = config.use_fp16 and (device.type == "cuda")

    if (
        _CACHED_TXT2IMG_PIPE is not None
        and _CACHED_PIPE_MODEL_ID == model_id
        and _CACHED_PIPE_DEVICE == device
    ):
        return _CACHED_TXT2IMG_PIPE

    # Clean existing memory
    cleanup_memory()

    try:
        from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    except ImportError as exc:
        raise PipelineExecutionError(
            "Missing 'diffusers' package. Please install diffusers with `pip install diffusers`."
        ) from exc

    torch_dtype = torch.float16 if (device.type == "cuda" and use_fp16) else torch.float32

    logger.info(f"Loading Text-to-Image pipeline: {model_id} on {device} (dtype={torch_dtype})...")

    try:
        pipeline_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
        }

        # Handle Hugging Face auth token if available
        if config.hf_token:
            pipeline_kwargs["token"] = config.hf_token

        # Attempt to load model pipeline
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, **pipeline_kwargs)
        except Exception:
            # Fallback to general DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model_id, **pipeline_kwargs)

        if device.type == "cuda":
            pipe = pipe.to(device)
            # Memory optimizations for consumer GPUs (e.g. RTX 2050 4GB)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()

            # For 4GB or lower VRAM, enable sequential or model CPU offload if available
            hw = detect_hardware()
            if hw.vram_total_mb > 0 and hw.vram_total_mb <= 4500:
                logger.info("Enabling model CPU offload for 4GB VRAM GPU...")
                try:
                    pipe.enable_model_cpu_offload()
                except Exception as offload_err:
                    logger.debug(f"enable_model_cpu_offload notice: {offload_err}")
        else:
            pipe = pipe.to(device)

        _CACHED_TXT2IMG_PIPE = pipe
        _CACHED_PIPE_MODEL_ID = model_id
        _CACHED_PIPE_DEVICE = device

        return _CACHED_TXT2IMG_PIPE

    except Exception as e:
        cleanup_memory()
        logger.error(f"Failed to load Text-to-Image pipeline: {e}")
        raise PipelineExecutionError(
            f"Unable to load model '{model_id}'. Ensure model weights are accessible and check your internet connection: {str(e)}"
        ) from e


def _call_hf_inference_api(
    prompt: str,
    negative_prompt: str = "",
    model_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Image.Image:
    """Call Hugging Face Serverless Inference API as clean fallback."""
    import io
    import requests

    if not token:
        raise PipelineExecutionError(
            "Hugging Face API fallback requested, but no HF_TOKEN was provided in .env or settings."
        )

    model = model_id or config.model_id
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
        },
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise PipelineExecutionError(
                f"Hugging Face API error ({response.status_code}): {response.text}"
            )
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as err:
        raise PipelineExecutionError(f"Hugging Face API fallback failed: {err}") from err


def generate_text_to_image(
    prompt: str,
    negative_prompt: str = "",
    style_preset: str = "None",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    device: Optional[torch.device] = None,
    model_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Image.Image, Path, Dict[str, Any]]:
    """Generate 512x512 image from text prompt with seed and style conditioning.

    Returns:
        Tuple containing (PIL Image, saved Path, metadata dict).
    """
    clean_prompt = validate_prompt(prompt)

    # Apply style preset prompt conditioning
    conditioned_prompt, conditioned_negative = apply_style_preset(
        base_prompt=clean_prompt,
        preset_name=style_preset,
        user_negative_prompt=negative_prompt,
    )

    if device is None:
        device = get_torch_device(config.device_mode)

    # Manage random seed
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    # Check for API Fallback mode if configured
    if config.hf_api_fallback and config.hf_token:
        logger.info("Using Hugging Face Inference API fallback as configured.")
        img = _call_hf_inference_api(
            prompt=conditioned_prompt,
            negative_prompt=conditioned_negative,
            model_id=model_id,
            token=config.hf_token,
        )
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    else:
        # Local generation pipeline
        try:
            pipe = load_text_to_image_pipeline(model_id=model_id, device=device)
        except Exception as load_err:
            if config.hf_token:
                logger.warning(f"Local loading failed ({load_err}). Attempting HF API fallback...")
                img = _call_hf_inference_api(
                    prompt=conditioned_prompt,
                    negative_prompt=conditioned_negative,
                    model_id=model_id,
                    token=config.hf_token,
                )
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            else:
                raise

        # Setup generator for reproducibility
        generator = torch.Generator(device=device if device.type == "cpu" else "cuda").manual_seed(seed)

        def step_callback(step: int, timestep: int, latents: Any):
            if progress_callback:
                progress_callback(step + 1, num_inference_steps)

        inference_kwargs: Dict[str, Any] = {
            "prompt": conditioned_prompt,
            "negative_prompt": conditioned_negative if conditioned_negative else None,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
        }

        # Add callback if supported
        try:
            inference_kwargs["callback"] = step_callback
            inference_kwargs["callback_steps"] = 1
        except Exception:
            pass

        try:
            with torch.inference_mode():
                output = pipe(**inference_kwargs)
            img = output.images[0]
        except torch.cuda.OutOfMemoryError as oom_err:
            cleanup_memory()
            logger.warning(f"CUDA OOM encountered during generation: {oom_err}. Retrying with CPU fallback...")
            # Fallback to CPU execution
            pipe_cpu = load_text_to_image_pipeline(model_id=model_id, device=torch.device("cpu"), use_fp16=False)
            gen_cpu = torch.Generator(device="cpu").manual_seed(seed)
            inference_kwargs["generator"] = gen_cpu
            with torch.inference_mode():
                output = pipe_cpu(**inference_kwargs)
            img = output.images[0]
        except Exception as gen_err:
            cleanup_memory()
            raise PipelineExecutionError(f"Inference execution failed: {str(gen_err)}") from gen_err

    # Metadata payload
    metadata: Dict[str, Any] = {
        "prompt": clean_prompt,
        "conditioned_prompt": conditioned_prompt,
        "negative_prompt": negative_prompt,
        "style_preset": style_preset,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "device": str(device),
        "model_id": model_id or config.model_id,
    }

    # Save to disk
    image_path, _ = save_image_with_metadata(
        image=img,
        output_dir=TEXT_TO_IMAGE_DIR,
        prefix="txt2img",
        metadata=metadata,
    )

    cleanup_memory()
    return img, image_path, metadata

