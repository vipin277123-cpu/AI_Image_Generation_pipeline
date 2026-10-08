"""Application configuration and settings management for AI Image Generation Studio."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


ROOT_DIR: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = ROOT_DIR / os.getenv("OUTPUTS_DIR", "outputs")
TEXT_TO_IMAGE_DIR: Path = OUTPUTS_DIR / "text_to_image"
STYLE_TRANSFER_DIR: Path = OUTPUTS_DIR / "style_transfer"
VARIATIONS_DIR: Path = OUTPUTS_DIR / "variations"

ASSETS_DIR: Path = ROOT_DIR / "assets"
STYLES_DIR: Path = ASSETS_DIR / "styles"
CONTENT_DIR: Path = ASSETS_DIR / "content"

# Ensure all essential directories exist
for directory in [
    OUTPUTS_DIR,
    TEXT_TO_IMAGE_DIR,
    STYLE_TRANSFER_DIR,
    VARIATIONS_DIR,
    ASSETS_DIR,
    STYLES_DIR,
    CONTENT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Preset prompt conditioning mappings
STYLE_PRESETS: Dict[str, Dict[str, str]] = {
    "None": {
        "prompt_suffix": "",
        "negative_prompt": "blurry, low quality, distorted, artifacts, bad anatomy",
    },
    "Realistic": {
        "prompt_suffix": ", photorealistic, 8k resolution, highly detailed, professional photography, natural lighting, sharp focus",
        "negative_prompt": "cartoon, 3d render, anime, painting, blurry, oversaturated, deformed, bad anatomy",
    },
    "Cinematic": {
        "prompt_suffix": ", cinematic still, 35mm photograph, dramatic film lighting, blockbuster movie aesthetic, shallow depth of field, anamorphic",
        "negative_prompt": "low budget, flat lighting, amateur, cartoon, drawing, distorted, oversaturated",
    },
    "Digital Art": {
        "prompt_suffix": ", digital art, trending on ArtStation, vivid colors, intricate details, smooth illustration, award-winning concept art",
        "negative_prompt": "grainy, low resolution, poorly drawn, photographic, noisy, flat",
    },
    "Anime": {
        "prompt_suffix": ", anime aesthetic, Makoto Shinkai style, studio anime key visual, expressive character design, vibrant anime color palette",
        "negative_prompt": "realistic photo, 3d realistic render, ugly, deformed eyes, extra limbs, dull colors",
    },
    "Fantasy": {
        "prompt_suffix": ", high fantasy, epic atmospheric lighting, magical glow, mythical aura, intricate fantasy illustration, masterpiece",
        "negative_prompt": "modern, futuristic, cars, skyscrapers, low resolution, blurry, grainy",
    },
    "Watercolor": {
        "prompt_suffix": ", delicate watercolor painting, soft pigment washes, paper texture, expressive brush strokes, flowing ink contours",
        "negative_prompt": "photograph, 3d render, digital gloss, sharp plastic, harsh edges",
    },
    "Oil Painting": {
        "prompt_suffix": ", classical oil painting, textured canvas, impasto brushwork, rich oil glaze, museum masterpiece, Rembrandt chiaroscuro",
        "negative_prompt": "digital vector, photo, flat cartoon, airbrushed, lowres, oversaturated plastic",
    },
    "Cyberpunk": {
        "prompt_suffix": ", cyberpunk theme, neon glow, holographic city night, chrome reflections, futuristic dystopia, cinematic rain wet asphalt",
        "negative_prompt": "vintage, medieval, pastel, sunlight pastoral, blurry, low contrast",
    },
}


@dataclass
class AppConfig:
    """Master application configuration container."""

    # Default Stable Diffusion Model (sd-turbo provides ultra-fast real-time inference on 4GB VRAM)
    model_id: str = os.getenv("DEFAULT_MODEL_ID", "stabilityai/sd-turbo")

    # Standard SD 1.5 alternative model
    standard_model_id: str = "runwayml/stable-diffusion-v1-5"

    # Device configuration: 'auto', 'cuda', 'cpu'
    device_mode: str = os.getenv("DEVICE", "auto").lower()

    # Precision
    use_fp16: bool = os.getenv("USE_FP16", "true").lower() in ("true", "1", "yes")

    # Low VRAM optimizations (attention slicing, cpu offloading)
    low_vram_mode: bool = os.getenv("LOW_VRAM_MODE", "true").lower() in ("true", "1", "yes")

    # Hugging Face API fallback
    hf_token: str | None = os.getenv("HF_TOKEN")
    hf_api_fallback: bool = os.getenv("HF_API_FALLBACK", "false").lower() in ("true", "1", "yes")

    # Default image parameters
    default_image_size: int = 512
    default_inference_steps: int = 25
    default_guidance_scale: float = 7.5
    default_strength: float = 0.75

    # Style Transfer defaults
    default_style_iterations: int = 150
    default_content_weight: float = 1.0
    default_style_weight: float = 1e5

    # Limits
    min_steps: int = 1
    max_steps: int = 75
    min_guidance: float = 1.0
    max_guidance: float = 20.0


# Global singleton instance
config = AppConfig()

