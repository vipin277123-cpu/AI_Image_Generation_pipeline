"""AI Image Generation Studio - Interactive Streamlit Web Interface."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from app.config import (
    CONTENT_DIR,
    STYLE_PRESETS,
    STYLES_DIR,
    config,
)
from app.hardware import cleanup_memory, detect_hardware, get_torch_device
from app.style_transfer import run_style_transfer
from app.text_to_image import generate_text_to_image, load_text_to_image_pipeline
from app.utils import (
    ImageStudioError,
    get_gallery_records,
    image_to_bytes,
    validate_and_load_image,
)
from app.variations import generate_image_variations, load_img2img_pipeline

# Configure application logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("studio_app")

# Page Configuration
st.set_page_config(
    page_title="AI Image Generation Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .hw-badge {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
    }
    .stDownloadButton button {
        width: 100%;
        border-radius: 6px;
    }
    .gallery-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px;
        background: #fafafa;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_txt2img_pipeline(model_id: str, device_mode: str, use_fp16: bool):
    """Cache Text-to-Image pipeline across Streamlit interactions."""
    device = get_torch_device(device_mode)
    return load_text_to_image_pipeline(model_id=model_id, device=device, use_fp16=use_fp16)


@st.cache_resource(show_spinner=False)
def get_cached_img2img_pipeline(model_id: str, device_mode: str, use_fp16: bool):
    """Cache Image-to-Image pipeline across Streamlit interactions."""
    device = get_torch_device(device_mode)
    return load_img2img_pipeline(model_id=model_id, device=device, use_fp16=use_fp16)


def render_sidebar():
    """Render hardware diagnostics and system settings in the sidebar."""
    st.sidebar.title("⚙️ System & Hardware")

    hw = detect_hardware(preferred_device_mode=config.device_mode)

    # Hardware Status Display
    st.sidebar.markdown("### 🖥️ Hardware Detection")
    sidebar_container = st.sidebar.container()

    if hw.cuda_available:
        sidebar_container.success("✅ **CUDA GPU Acceleration Active**")
    elif hw.gpu_available:
        sidebar_container.warning("⚠️ **GPU Found (CPU PyTorch Active)**")
    else:
        sidebar_container.info("ℹ️ **CPU Execution Mode**")

    sidebar_container.markdown(f"**CPU:** {hw.cpu_name}")
    sidebar_container.markdown(f"**Cores:** {hw.cpu_cores_physical} physical / {hw.cpu_cores_logical} logical")
    sidebar_container.markdown(f"**RAM:** {hw.total_ram_gb:.1f} GB")
    sidebar_container.markdown(f"**GPU:** {hw.gpu_name}")
    sidebar_container.markdown(f"**CUDA:** {'Available' if hw.cuda_available else 'Unavailable'}")

    if hw.gpu_available and hw.vram_total_mb > 0:
        sidebar_container.markdown(f"**VRAM Total:** {hw.vram_total_mb:.0f} MB")
        if hw.cuda_available:
            sidebar_container.markdown(f"**VRAM Free:** {hw.vram_free_mb:.0f} MB")

    sidebar_container.markdown(f"**Active Device:** `{hw.active_device}`")

    if hw.warning:
        sidebar_container.warning(hw.warning)

    st.sidebar.divider()

    # Memory Management
    st.sidebar.markdown("### 🧹 Memory Tools")
    if st.sidebar.button("Clear Cache & Memory", use_container_width=True):
        cleanup_memory()
        st.sidebar.success("Memory cleared successfully!")

    st.sidebar.divider()

    # Pipeline Model Configuration
    st.sidebar.markdown("### 🤖 Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Diffusion Model",
        options=[
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/sd-turbo",
            "CompVis/stable-diffusion-v1-4",
        ],
        index=0,
        help="Model checkpoint for Text-to-Image and Image Variations.",
    )
    config.model_id = model_choice

    # API Fallback status
    if config.hf_api_fallback:
        st.sidebar.info("🌐 HF Serverless API Fallback is enabled.")

    st.sidebar.caption("AI Image Generation Studio v1.0.0")


def render_text_to_image_tab():
    """Render Text-to-Image generation tab."""
    st.markdown("### 🎨 Text-to-Image Generation")
    st.caption("Generate high-resolution 512x512 imagery from descriptive natural language prompts.")

    col_input, col_preview = st.columns([1, 1], gap="large")

    with col_input:
        prompt = st.text_area(
            "Prompt",
            value="A futuristic cyberpunk city at night with flying cars, neon billboards, cinematic rain reflections",
            height=110,
            placeholder="Describe what you want to see...",
            help="Primary description of the image content.",
        )

        negative_prompt = st.text_input(
            "Negative Prompt (Optional)",
            value="blurry, distorted, ugly, low quality, artifacts",
            help="Elements you wish to exclude from the generated image.",
        )

        preset_names = list(STYLE_PRESETS.keys())
        style_preset = st.selectbox(
            "Style Preset",
            options=preset_names,
            index=preset_names.index("Cyberpunk") if "Cyberpunk" in preset_names else 0,
            help="Conditions the prompt with specialized aesthetic descriptors.",
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            num_steps = st.slider("Inference Steps", min_value=1, max_value=60, value=25, step=1)
            seed_input = st.number_input(
                "Seed (-1 for random)",
                min_value=-1,
                max_value=2**32 - 1,
                value=-1,
                step=1,
            )
        with col_s2:
            guidance_scale = st.slider("Guidance Scale (CFG)", min_value=1.0, max_value=18.0, value=7.5, step=0.5)

        generate_clicked = st.button("✨ Generate Image", type="primary", use_container_width=True)

    with col_preview:
        if generate_clicked:
            if not prompt.strip():
                st.error("Please enter a valid text prompt before generating.")
                return

            seed_val = None if seed_input < 0 else int(seed_input)
            progress_bar = st.progress(0, text="Initializing generation...")

            try:
                def on_progress(step: int, total: int):
                    progress_bar.progress(int((step / total) * 100), text=f"Denoising step {step}/{total}...")

                with st.spinner("Generating image..."):
                    img, path, meta = generate_text_to_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        style_preset=style_preset,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        seed=seed_val,
                        progress_callback=on_progress,
                    )

                progress_bar.progress(100, text="Generation complete!")
                st.image(img, caption=f"Prompt: {prompt[:60]}...", use_container_width=True)

                # Parameter summary
                with st.expander("Generation Parameters", expanded=False):
                    st.json(meta)

                # Download button
                img_bytes = image_to_bytes(img)
                st.download_button(
                    label="💾 Download Image (PNG)",
                    data=img_bytes,
                    file_name=f"txt2img_{meta.get('seed', 'output')}.png",
                    mime="image/png",
                    use_container_width=True,
                )

            except ImageStudioError as ise:
                progress_bar.empty()
                st.error(f"Generation error: {str(ise)}")
            except Exception as ex:
                progress_bar.empty()
                logger.exception("Unexpected error in text-to-image generation")
                st.error(f"An unexpected error occurred during generation: {str(ex)}")
        else:
            st.info("Enter your prompt and click **Generate Image** to preview the output here.")


def render_style_transfer_tab():
    """Render VGG19 Neural Style Transfer tab."""
    st.markdown("### 🖌️ Neural Style Transfer (VGG19)")
    st.caption("Transfer artistic texture, colors, and brushwork from a style image onto a content image.")

    col_setup, col_result = st.columns([1, 1], gap="large")

    with col_setup:
        st.markdown("#### 1. Input Images")
        col_c, col_s = st.columns(2)

        with col_c:
            content_file = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg", "webp"])
            # Provide sample content if none uploaded
            sample_content_files = list(CONTENT_DIR.glob("*.jpg")) + list(CONTENT_DIR.glob("*.png"))
            use_sample_content = False
            if not content_file and sample_content_files:
                selected_sample_c = st.selectbox(
                    "Or choose sample content:",
                    options=[f.name for f in sample_content_files],
                )
                use_sample_content = True

        with col_s:
            style_file = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg", "webp"])
            # Provide predefined styles from assets/styles/
            sample_style_files = list(STYLES_DIR.glob("*.jpg")) + list(STYLES_DIR.glob("*.png"))
            use_sample_style = False
            if not style_file and sample_style_files:
                selected_sample_s = st.selectbox(
                    "Or choose predefined style:",
                    options=[f.name for f in sample_style_files],
                )
                use_sample_style = True

        st.markdown("#### 2. Optimization Parameters")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            iterations = st.slider("Optimization Iterations", min_value=20, max_value=300, value=120, step=10)
            img_res = st.select_slider("Resolution", options=[256, 384, 512], value=384)
        with col_p2:
            content_weight = st.number_input("Content Weight", min_value=0.1, max_value=50.0, value=1.0, step=0.5)
            style_weight = st.number_input("Style Weight", min_value=1e3, max_value=1e8, value=1e5, step=1e4, format="%.0f")

        transfer_clicked = st.button("🎨 Run Neural Style Transfer", type="primary", use_container_width=True)

    with col_result:
        # Load content and style images
        content_img: Optional[Image.Image] = None
        style_img: Optional[Image.Image] = None

        if content_file:
            content_img = validate_and_load_image(content_file)
        elif use_sample_content and sample_content_files:
            c_path = CONTENT_DIR / selected_sample_c
            content_img = validate_and_load_image(c_path)

        if style_file:
            style_img = validate_and_load_image(style_file)
        elif use_sample_style and sample_style_files:
            s_path = STYLES_DIR / selected_sample_s
            style_img = validate_and_load_image(s_path)

        # Previews
        if content_img and style_img:
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.image(content_img, caption="Content Image", use_container_width=True)
            with p_col2:
                st.image(style_img, caption="Style Reference", use_container_width=True)

        if transfer_clicked:
            if not content_img or not style_img:
                st.error("Please provide both a content image and a style image.")
                return

            prog_bar = st.progress(0, text="Initializing VGG19 optimization...")

            try:
                def on_style_progress(step: int, total: int, loss: float):
                    percent = int((step / total) * 100)
                    prog_bar.progress(percent, text=f"Optimizing style: step {step}/{total} (loss: {loss:.2f})")

                with st.spinner("Stylizing image with VGG19..."):
                    stylized = run_style_transfer(
                        content_image=content_img,
                        style_image=style_img,
                        num_iterations=iterations,
                        content_weight=float(content_weight),
                        style_weight=float(style_weight),
                        image_size=img_res,
                        progress_callback=on_style_progress,
                    )

                prog_bar.progress(100, text="Style transfer completed!")
                st.image(stylized, caption="Stylized Result", use_container_width=True)

                # Save metadata and file
                from app.config import STYLE_TRANSFER_DIR
                from app.utils import save_image_with_metadata

                saved_p, _ = save_image_with_metadata(
                    image=stylized,
                    output_dir=STYLE_TRANSFER_DIR,
                    prefix="style_transfer",
                    metadata={
                        "iterations": iterations,
                        "content_weight": content_weight,
                        "style_weight": style_weight,
                        "resolution": img_res,
                    },
                )

                st.download_button(
                    label="💾 Download Stylized Image",
                    data=image_to_bytes(stylized),
                    file_name=saved_p.name,
                    mime="image/png",
                    use_container_width=True,
                )

            except ImageStudioError as ise:
                prog_bar.empty()
                st.error(f"Style Transfer error: {str(ise)}")
            except Exception as ex:
                prog_bar.empty()
                logger.exception("Unexpected error in style transfer")
                st.error(f"An unexpected error occurred during style transfer: {str(ex)}")
        elif not (content_img and style_img):
            st.info("Upload content and style images to preview and generate stylized artwork.")


def render_variations_tab():
    """Render Image Variations tab producing exactly 4 variations."""
    st.markdown("### 🔀 Image Variations")
    st.caption("Generate exactly 4 creative variations from an uploaded image using different stochastic seeds.")

    col_setup, col_result = st.columns([1, 1], gap="large")

    with col_setup:
        var_file = st.file_uploader("Upload Source Image", type=["png", "jpg", "jpeg", "webp"])

        # Sample images fallback
        sample_content_files = list(CONTENT_DIR.glob("*.jpg")) + list(CONTENT_DIR.glob("*.png"))
        use_sample = False
        if not var_file and sample_content_files:
            sample_choice = st.selectbox("Or choose sample image:", options=[f.name for f in sample_content_files])
            use_sample = True

        prompt_guidance = st.text_input(
            "Variation Prompt (Optional)",
            value="",
            placeholder="e.g. vibrant colors, dramatic lighting, detailed texture",
            help="Optional guiding prompt for the variations.",
        )

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            strength = st.slider(
                "Transformation Strength",
                min_value=0.2,
                max_value=0.95,
                value=0.75,
                step=0.05,
                help="Higher values produce more divergence from original image.",
            )
            base_seed = st.number_input(
                "Base Seed (-1 for random)",
                min_value=-1,
                max_value=2**32 - 1,
                value=-1,
            )
        with col_v2:
            num_steps = st.slider("Inference Steps", min_value=5, max_value=50, value=25, step=1)
            guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5)

        generate_var_clicked = st.button("🔀 Generate 4 Variations", type="primary", use_container_width=True)

    with col_result:
        source_image: Optional[Image.Image] = None
        if var_file:
            source_image = validate_and_load_image(var_file)
        elif use_sample and sample_content_files:
            source_image = validate_and_load_image(CONTENT_DIR / sample_choice)

        if source_image:
            st.image(source_image, caption="Input Source Image", width=250)

        if generate_var_clicked:
            if not source_image:
                st.error("Please upload an input image first.")
                return

            seed_val = None if base_seed < 0 else int(base_seed)
            prog_var = st.progress(0, text="Generating 4 variations...")

            try:
                def on_var_progress(current: int, total: int):
                    prog_var.progress(int((current / total) * 100), text=f"Finished variation {current} of {total}...")

                with st.spinner("Synthesizing 4 variations..."):
                    results = generate_image_variations(
                        input_image=source_image,
                        num_variations=4,
                        strength=strength,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        prompt=prompt_guidance,
                        base_seed=seed_val,
                        progress_callback=on_var_progress,
                    )

                prog_var.progress(100, text="All 4 variations successfully generated!")

                st.markdown("#### Generated Variations (4 Distinct Seeds)")
                grid_c1, grid_c2 = st.columns(2)

                for idx, (var_img, path, meta) in enumerate(results):
                    target_col = grid_c1 if idx % 2 == 0 else grid_c2
                    with target_col:
                        st.image(var_img, caption=f"Variation {idx + 1} (Seed: {meta['seed']})", use_container_width=True)
                        st.download_button(
                            label=f"💾 Download #{idx + 1}",
                            data=image_to_bytes(var_img),
                            file_name=path.name,
                            mime="image/png",
                            key=f"dl_var_{idx}_{path.name}",
                            use_container_width=True,
                        )

            except ImageStudioError as ise:
                prog_var.empty()
                st.error(f"Variations error: {str(ise)}")
            except Exception as ex:
                prog_var.empty()
                logger.exception("Unexpected error in image variations")
                st.error(f"An unexpected error occurred during variations: {str(ex)}")
        elif not source_image:
            st.info("Upload an image to see its 4 distinct variations.")


def render_gallery_tab():
    """Render browsable gallery of generated images with download capabilities."""
    st.markdown("### 🖼️ Generated Images Gallery")
    st.caption("Browse, inspect parameters, and download previously generated images.")

    filter_choice = st.selectbox(
        "Filter by Pipeline",
        options=["All", "Text-to-Image", "Style Transfer", "Variations"],
        index=0,
    )

    cat_map = {
        "All": None,
        "Text-to-Image": "text_to_image",
        "Style Transfer": "style_transfer",
        "Variations": "variations",
    }

    records = get_gallery_records(category=cat_map[filter_choice], limit=60)

    if not records:
        st.info("No generated images found yet. Create some artworks in the generation tabs above!")
        return

    st.write(f"Showing **{len(records)}** generated items:")

    # Render gallery grid (3 columns)
    cols = st.columns(3)
    for idx, item in enumerate(records):
        col = cols[idx % 3]
        with col:
            img_path = Path(item["path"])
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    caption_text = f"**{item['category']}**"
                    if item.get("prompt"):
                        caption_text += f"\n_{item['prompt'][:50]}..._"
                    st.markdown(caption_text)

                    with st.expander("Parameters", expanded=False):
                        st.json(item.get("parameters", {}))

                    st.download_button(
                        label="💾 Download",
                        data=image_to_bytes(img),
                        file_name=img_path.name,
                        mime="image/png",
                        key=f"gal_dl_{idx}_{img_path.name}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Failed to display {img_path.name}: {e}")


def main():
    """Application entrypoint."""
    st.markdown('<div class="main-header">AI Image Generation Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-modal AI art creation with Stable Diffusion & VGG19 Neural Style Transfer</div>',
        unsafe_allow_html=True,
    )

    # Render Sidebar diagnostics
    render_sidebar()

    # Main Navigation Tabs
    tab_t2i, tab_style, tab_var, tab_gal = st.tabs([
        "🎨 Text-to-Image",
        "🖌️ Style Transfer",
        "🔀 Image Variations (4x)",
        "🖼️ Gallery",
    ])

    with tab_t2i:
        render_text_to_image_tab()

    with tab_style:
        render_style_transfer_tab()

    with tab_var:
        render_variations_tab()

    with tab_gal:
        render_gallery_tab()


if __name__ == "__main__":
    main()

