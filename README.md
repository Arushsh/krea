---
title: Krea 2 Turbo Image Generator
emoji: 🖌️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.44.1
python_version: '3.12'
app_file: app.py
pinned: true
short_description: Krea 2 Turbo text2image and image editing
license: other
---

# Krea 2 Turbo Image Generator

This Space runs Krea 2 Turbo through headless ComfyUI execution. It provides:

- Text-to-image generation.
- Instruction-based image editing with an optional second reference image.
- Krea 2 Identity Edit v1.2 conditioning for the edit mode.
- Seed, sampling, grounding, and reference-fidelity controls.
- PNG metadata and portable JSON settings profiles.

The Space intentionally does **not** expose optional style LoRAs or arbitrary LoRA uploads. The
identity-edit adapter is a required implementation detail of the edit workflow and is always used
in that mode.

## Models

The text encoder and VAE are downloaded from the ungated [Comfy-Org/Krea-2 mirror](https://huggingface.co/Comfy-Org/Krea-2).
The Space uses the custom [PornMaster Krea 2 Turbo checkpoint](https://huggingface.co/mpasila/Krea-2-Models/resolve/main/pornmasterKrea2_v2TurboInt8.safetensors)
instead of the official Krea 2 Turbo diffusion checkpoint:

| File | Role |
|------|------|
| `pornmasterKrea2_v2TurboInt8.safetensors` | PornMaster Krea 2 Turbo diffusion model |
| `qwen3vl_4b_fp8_scaled.safetensors` | Krea 2 Qwen3-VL text encoder |
| `qwen_image_vae.safetensors` | Krea 2 VAE |

Edit mode also downloads `krea2_identity_edit_v1_2.safetensors` from
[conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit).
The edit nodes come from [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit).

All model and adapter licenses remain applicable. Review the Krea 2 Community License Agreement
and the adapter repository’s responsible-use guidance before deploying this Space. Do not use the
identity-edit workflow for non-consensual or harmful impersonation of real people.

## Usage

Choose **text-to-image** for a new image, or **edit** to provide a primary source image and an
optional second reference. In edit mode, image order matters: the primary image is the scene/source
and the second image is the subject or additional reference.

The edit workflow uses `fit` reference geometry. The target megapixel setting controls the output
size while preserving the primary image’s aspect ratio. Lower grounding values usually follow the
edit instruction more strongly; higher values usually preserve reference identity more strongly.

Generated PNGs include a `krea_settings` JSON object and a readable `parameters` field. Source image
bytes are never stored in exported settings profiles.
