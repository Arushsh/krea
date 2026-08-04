"""Krea 2 Turbo text-to-image and image-editing Hugging Face Space.

The supplied ComfyUI workflows are bundled as the graph references for this
Space.  The runtime uses their supported first-pass/edit paths and replaces
optional UI-heavy stages with a small deterministic API graph so startup does
not depend on face, hand, detailer, or image-saver node packs.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

try:
    import spaces
except ImportError:  # pragma: no cover - lets pure unit tests import helpers.
    class _SpacesFallback:
        @staticmethod
        def GPU(**_kwargs):
            def decorate(function):
                return function
            return decorate

    spaces = _SpacesFallback()

from settings_utils import (
    APP_ID,
    PROFILE_SCHEMA_VERSION,
    build_settings,
    extract_image_settings,
    parse_settings_text,
    write_png_metadata,
)


ROOT = pathlib.Path(__file__).resolve().parent
COMFY = ROOT / "ComfyUI"
MODELS = COMFY / "models"
INPUT = COMFY / "input"
OUTPUT = COMFY / "output"
CUSTOM_NODES = COMFY / "custom_nodes"

T2I_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2.json"
EDIT_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2Edit.json"

KREA_REPO = "Comfy-Org/Krea-2"
IDENTITY_REPO = "conradlocke/krea2-identity-edit"
IDENTITY_FILE = "krea2_identity_edit_v1_2.safetensors"
KREA_EDIT_NODES = "https://github.com/lbouaraba/comfyui-krea2edit.git"

DOWNLOADS = [
    (KREA_REPO, "diffusion_models/krea2_turbo_fp8_scaled.safetensors", MODELS / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors", "Krea 2 Turbo diffusion model"),
    (KREA_REPO, "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", MODELS / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors", "Qwen3-VL text encoder"),
    (KREA_REPO, "vae/qwen_image_vae.safetensors", MODELS / "vae" / "qwen_image_vae.safetensors", "Qwen image VAE"),
    (IDENTITY_REPO, IDENTITY_FILE, MODELS / "loras" / "krea" / IDENTITY_FILE, "Krea 2 identity-edit adapter"),
]

SAMPLERS = [
    "euler", "euler_ancestral", "euler_a", "dpmpp_2m", "dpmpp_2m_sde",
    "dpmpp_sde", "heun", "lms",
]
SCHEDULERS = ["beta", "normal", "karras", "exponential", "sgm_uniform", "simple"]

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_TARGET_MP = 1.4
DEFAULT_GROUNDING = 768
DEFAULT_REF_BOOST = 1.0
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "beta"
DEFAULT_SEED = 2
MIN_GPU_SECONDS = int(os.environ.get("MIN_GPU_SECONDS", "45"))
MAX_GPU_SECONDS = int(os.environ.get("MAX_GPU_SECONDS", "300"))

_comfy_ready = False
_nodes_ready = False
_workflow_cache: dict[str, dict[str, Any]] = {}


def _run(command: list[str], cwd: pathlib.Path | None = None, check: bool = True) -> None:
    print("[setup]", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=check)


def _pip_install(arguments: list[str]) -> None:
    _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *arguments], check=False)


def _install_filtered_requirements(path: pathlib.Path) -> None:
    """Install ComfyUI requirements without replacing Space's torch stack."""
    if not path.exists():
        return
    blocked = {"torch", "torchvision", "torchaudio", "transformers", "huggingface-hub", "accelerate"}
    requirements: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        package = re.split(r"[<>=!~;\[\s]", item.lower().replace("_", "-"), maxsplit=1)[0]
        if package not in blocked:
            requirements.append(item)
    if requirements:
        _pip_install(requirements)


def _ensure_repo(path: pathlib.Path, url: str) -> None:
    if not path.exists():
        _run(["git", "clone", "--depth", "1", url, str(path)])


def _restore_utils_namespace() -> None:
    """Undo the old root ``utils`` rename if a persistent Space has one."""
    source = COMFY / "utils"
    target = COMFY / "utilities"
    if not source.exists() and target.exists():
        target.rename(source)
    if not source.exists():
        return
    for path in COMFY.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = re.sub(r"\bfrom utilities\b", "from utils", text)
        updated = re.sub(r"\bimport utilities\b", "import utils", updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _ensure_comfy() -> None:
    global _comfy_ready
    if _comfy_ready:
        return
    _ensure_repo(COMFY, "https://github.com/comfyanonymous/ComfyUI.git")
    _install_filtered_requirements(COMFY / "requirements.txt")
    CUSTOM_NODES.mkdir(parents=True, exist_ok=True)
    _ensure_repo(CUSTOM_NODES / "comfyui-krea2edit", KREA_EDIT_NODES)
    _restore_utils_namespace()
    for folder in ("diffusion_models", "text_encoders", "vae", "loras/krea"):
        (MODELS / folder).mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _comfy_ready = True


def _init_comfy_nodes() -> None:
    global _nodes_ready
    if _nodes_ready:
        return
    comfy_path = str(COMFY)
    sys.path = [item for item in sys.path if item != comfy_path]
    sys.path.insert(0, comfy_path)
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    os.chdir(COMFY)
    import execution
    import nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    loop.run_until_complete(nodes.init_extra_nodes())
    _nodes_ready = True


def _download_to_dest(repo: str, filename: str, destination: pathlib.Path, label: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(filename)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=repo,
            filename=path.name,
            subfolder=None if str(path.parent) == "." else str(path.parent),
            local_dir=str(destination.parent),
            token=token,
        )
    )
    if downloaded.resolve() != destination.resolve():
        shutil.move(str(downloaded), str(destination))
    print(f"[models] ready: {label}", flush=True)


def _ensure_models(progress: gr.Progress | None = None) -> None:
    for index, (repo, filename, destination, label) in enumerate(DOWNLOADS):
        if progress:
            progress(index / len(DOWNLOADS), desc=f"downloading {label}")
        _download_to_dest(repo, filename, destination, label)


def _read_source_workflow(path: pathlib.Path) -> dict[str, Any]:
    """Load a bundled visual workflow and fail early if it was not shipped."""
    if not path.exists():
        raise FileNotFoundError(f"workflow file is missing: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("nodes"):
        raise ValueError(f"workflow file has no nodes: {path.name}")
    return data


def _ref(node: str, output: int = 0) -> list[Any]:
    return [node, output]


def _t2i_workflow() -> dict[str, Any]:
    if "text2image" in _workflow_cache:
        return json.loads(json.dumps(_workflow_cache["text2image"]))
    _read_source_workflow(T2I_SOURCE)
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("1"), "shift": 4.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("2"), "text": ""}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("5")}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": _ref("4"), "positive": _ref("5"), "negative": _ref("6"), "latent_image": _ref("7"), "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG, "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER, "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": _ref("8"), "vae": _ref("3")}},
        "10": {"class_type": "SaveImage", "inputs": {"images": _ref("9"), "filename_prefix": "krea2_turbo"}},
    }
    _workflow_cache["text2image"] = workflow
    return json.loads(json.dumps(workflow))


def _edit_workflow(has_second_reference: bool) -> dict[str, Any]:
    _read_source_workflow(EDIT_SOURCE)
    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": _ref("5"), "lora_name": "krea/" + IDENTITY_FILE, "strength_model": 1.0}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": _ref("1"), "vae": _ref("4")}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "9": {"class_type": "Krea2EditModelPatch", "inputs": {"model": _ref("6"), "source_latent": _ref("7"), "vae": _ref("4"), "source_image": _ref("1"), "target_latent": _ref("8"), "ref_boost": DEFAULT_REF_BOOST, "ref_boost_a": DEFAULT_REF_BOOST, "fit_mode": "fit"}},
        "10": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": _ref("3"), "image": _ref("1"), "prompt": "", "grounding_px": DEFAULT_GROUNDING}},
        "11": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": _ref("10")}},
        "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("9"), "shift": 4.0}},
        "13": {"class_type": "KSampler", "inputs": {"model": _ref("12"), "positive": _ref("10"), "negative": _ref("11"), "latent_image": _ref("8"), "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG, "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER, "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": _ref("13"), "vae": _ref("4")}},
        "15": {"class_type": "SaveImage", "inputs": {"images": _ref("14"), "filename_prefix": "krea2_edit"}},
    }
    if has_second_reference:
        workflow["2"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
        workflow["16"] = {"class_type": "VAEEncode", "inputs": {"pixels": _ref("2"), "vae": _ref("4")}}
        workflow["9"]["inputs"]["source_latent_b"] = _ref("16")
        workflow["9"]["inputs"]["source_image_b"] = _ref("2")
        workflow["10"]["inputs"]["image_b"] = _ref("2")
    return workflow


def _find_node(workflow: dict[str, Any], class_type: str) -> str:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id
    raise KeyError(f"workflow does not contain {class_type}")


def _prepare_edit_image(path: str, target_megapixels: float) -> tuple[str, int, int]:
    """Resize the primary source to a 64-pixel grid while preserving its AR."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        megapixels = max(0.25, min(2.0, float(target_megapixels)))
        scale = (megapixels * 1_000_000 / max(1, image.width * image.height)) ** 0.5
        width = max(64, int(round(image.width * scale / 64) * 64))
        height = max(64, int(round(image.height * scale / 64) * 64))
        width = min(2048, width)
        height = min(2048, height)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        name = f"input_{uuid.uuid4().hex[:12]}.png"
        image.save(INPUT / name, format="PNG")
    return name, width, height


def _stage_image(path: str, prefix: str) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        name = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
        image.save(INPUT / name, format="PNG")
    return name


def _validate_request(mode: str, prompt: str, edit_prompt: str, primary: str | None) -> None:
    if mode not in {"text2image", "edit"}:
        raise ValueError("unsupported generation mode")
    if mode == "text2image" and not (prompt or "").strip():
        raise ValueError("enter a prompt")
    if mode == "edit":
        if not primary:
            raise ValueError("upload a primary image for edit mode")
        if not (edit_prompt or "").strip():
            raise ValueError("enter an edit instruction")


def _inject_t2i(workflow: dict[str, Any], *, prompt: str, width: int, height: int, steps: int, cfg: float, sampler: str, scheduler: str, seed: int) -> None:
    workflow["5"]["inputs"]["text"] = prompt.strip()
    workflow["7"]["inputs"].update(width=int(width), height=int(height))
    workflow["8"]["inputs"].update(seed=int(seed), steps=int(steps), cfg=float(cfg), sampler_name=sampler, scheduler=scheduler, denoise=1.0)


def _inject_edit(workflow: dict[str, Any], *, primary_name: str, second_name: str | None, width: int, height: int, edit_prompt: str, grounding_px: int, ref_boost: float, ref_boost_a: float, steps: int, cfg: float, sampler: str, scheduler: str, seed: int) -> None:
    workflow["1"]["inputs"]["image"] = primary_name
    workflow["8"]["inputs"].update(width=int(width), height=int(height))
    workflow["9"]["inputs"].update(ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a))
    workflow["10"]["inputs"].update(prompt=edit_prompt.strip(), grounding_px=int(grounding_px))
    workflow["13"]["inputs"].update(seed=int(seed), steps=int(steps), cfg=float(cfg), sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    if second_name:
        workflow["2"]["inputs"]["image"] = second_name


def _execute_workflow(workflow: dict[str, Any]) -> list[str]:
    import execution
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    executor = execution.PromptExecutor(
        server_instance,
        cache_type=execution.CacheType.RAM_PRESSURE,
        cache_args={"lru": 0, "ram": 2.0, "ram_inactive": 8.0},
    )
    prompt_id = str(uuid.uuid4())
    save_id = _find_node(workflow, "SaveImage")
    executor.execute(workflow, prompt_id, extra_data={}, execute_outputs=[save_id])
    if not executor.success:
        message = executor.status_messages[-1] if executor.status_messages else "ComfyUI execution failed"
        raise RuntimeError(str(message))
    paths: list[pathlib.Path] = []
    for output in executor.history_result.get("outputs", {}).values():
        for items in output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                base = OUTPUT if item.get("type", "output") == "output" else COMFY / item.get("type", "output")
                candidate = base / item.get("subfolder", "") / item["filename"]
                if candidate.exists():
                    paths.append(candidate)
    if not paths:
        paths = sorted(
            [pathlib.Path(item) for item in glob.glob(str(OUTPUT / "**" / "*.png"), recursive=True)],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    if not paths:
        raise RuntimeError("ComfyUI finished without an output image")
    return [str(path) for path in paths]


def _prepare_runtime(progress: gr.Progress | None = None) -> None:
    _ensure_comfy()
    _ensure_models(progress)
    _init_comfy_nodes()


def get_gpu_duration(*args: Any, **kwargs: Any) -> int:
    """Estimate GPU time for Spaces' positional or keyword callback forms."""
    # spaces.zero invokes a duration callback with the complete generate()
    # argument list positionally. Keep the indexes aligned with generate().
    steps = kwargs.get("steps", args[11] if len(args) > 11 else DEFAULT_STEPS)
    width = kwargs.get("width", args[5] if len(args) > 5 else DEFAULT_WIDTH)
    height = kwargs.get("height", args[6] if len(args) > 6 else DEFAULT_HEIGHT)
    gen_budget = kwargs.get("gen_budget", args[17] if len(args) > 17 else 0)
    if gen_budget and int(gen_budget) > 0:
        return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, int(gen_budget)))
    estimate = int(35 + (int(width) * int(height) / 1_000_000) * int(steps) * 3.0)
    return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, estimate))


@spaces.GPU(duration=get_gpu_duration)
def generate(
    mode: str,
    prompt: str,
    edit_prompt: str,
    primary_image: str | None,
    second_image: str | None,
    width: int,
    height: int,
    target_megapixels: float,
    grounding_px: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    randomize_seed: bool,
    gen_budget: float,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[list[str], str, int]:
    """Validate inputs, execute the selected Krea graph, and persist outputs."""
    effective_seed = random.randint(0, 2**32 - 1) if randomize_seed or int(seed) < 0 else int(seed)
    staged: list[pathlib.Path] = []
    try:
        _validate_request(mode, prompt, edit_prompt, primary_image)
        if sampler not in SAMPLERS or scheduler not in SCHEDULERS:
            raise ValueError("unsupported sampler or scheduler")
        _prepare_runtime(progress)
        if mode == "text2image":
            width = max(512, min(1536, int(width) // 64 * 64))
            height = max(512, min(1536, int(height) // 64 * 64))
            workflow = _t2i_workflow()
            _inject_t2i(workflow, prompt=prompt, width=width, height=height, steps=int(steps), cfg=float(cfg), sampler=sampler, scheduler=scheduler, seed=effective_seed)
        else:
            primary_name, width, height = _prepare_edit_image(primary_image, target_megapixels)
            staged.append(INPUT / primary_name)
            second_name = _stage_image(second_image, "reference") if second_image else None
            if second_name:
                staged.append(INPUT / second_name)
            workflow = _edit_workflow(bool(second_name))
            _inject_edit(workflow, primary_name=primary_name, second_name=second_name, width=width, height=height, edit_prompt=edit_prompt, grounding_px=int(grounding_px), ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a), steps=int(steps), cfg=float(cfg), sampler=sampler, scheduler=scheduler, seed=effective_seed)

        settings = build_settings(mode=mode, prompt=prompt, edit_prompt=edit_prompt, width=width, height=height, target_megapixels=float(target_megapixels), grounding_px=int(grounding_px), ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a), steps=int(steps), cfg=float(cfg), sampler_name=sampler, scheduler=scheduler, seed=int(seed), randomize_seed=bool(randomize_seed), gen_budget=float(gen_budget), effective_seed=effective_seed)
        progress(0.35, desc=f"generating {mode}")
        result_paths = _execute_workflow(workflow)
        destination_dir = pathlib.Path(tempfile.mkdtemp(prefix="krea2_outputs_"))
        output_paths: list[str] = []
        for index, source in enumerate(result_paths):
            destination = destination_dir / f"output_{index}.png"
            write_png_metadata(source, destination, settings)
            output_paths.append(str(destination))
        return output_paths, f"done — {len(output_paths)} image(s), seed {effective_seed}", effective_seed
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        raise gr.Error(f"generation failed: {str(exc)[:500]}") from exc
    finally:
        for path in staged:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _profile_from_values(mode: str, prompt: str, edit_prompt: str, width: int, height: int, target_mp: float, grounding: int, ref_boost: float, ref_boost_a: float, steps: int, cfg: float, sampler: str, scheduler: str, seed: int, randomize: bool, budget: float) -> dict[str, Any]:
    return build_settings(mode=mode, prompt=prompt, edit_prompt=edit_prompt, width=int(width), height=int(height), target_megapixels=float(target_mp), grounding_px=int(grounding), ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a), steps=int(steps), cfg=float(cfg), sampler_name=sampler, scheduler=scheduler, seed=int(seed), randomize_seed=bool(randomize), gen_budget=float(budget))


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Krea 2 Turbo Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖌️ Krea 2 Turbo Image Generator\nGenerate new images or edit a source image with Krea 2 Identity Edit.")
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["text2image", "edit"], value="text2image", label="mode")
                with gr.Column(visible=False) as image_inputs:
                    primary = gr.Image(type="filepath", label="primary image / scene")
                    second = gr.Image(type="filepath", label="optional second reference")
                prompt = gr.Textbox(value="A cinematic portrait in soft natural light", label="prompt", lines=3)
                edit_prompt = gr.Textbox(label="edit instruction", lines=3, visible=False, placeholder="recolor the jacket to matte black")
                with gr.Column() as t2i_resolution:
                    with gr.Row():
                        width = gr.Slider(512, 1536, value=DEFAULT_WIDTH, step=64, label="width")
                        height = gr.Slider(512, 1536, value=DEFAULT_HEIGHT, step=64, label="height")
                with gr.Column(visible=False) as edit_controls:
                    target_mp = gr.Slider(0.25, 2.0, value=DEFAULT_TARGET_MP, step=0.05, label="target megapixels")
                    grounding = gr.Slider(384, 1536, value=DEFAULT_GROUNDING, step=32, label="grounding resolution")
                    ref_boost = gr.Slider(0.0, 12.0, value=DEFAULT_REF_BOOST, step=0.1, label="primary reference strength")
                    ref_boost_a = gr.Slider(0.0, 12.0, value=DEFAULT_REF_BOOST, step=0.1, label="second reference strength")
                with gr.Accordion("sampling", open=False):
                    steps = gr.Slider(4, 40, value=DEFAULT_STEPS, step=1, label="steps")
                    cfg = gr.Slider(1.0, 5.0, value=DEFAULT_CFG, step=0.1, label="CFG")
                    with gr.Row():
                        sampler = gr.Dropdown(SAMPLERS, value=DEFAULT_SAMPLER, label="sampler")
                        scheduler = gr.Dropdown(SCHEDULERS, value=DEFAULT_SCHEDULER, label="scheduler")
                with gr.Row():
                    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="seed")
                    randomize = gr.Checkbox(value=False, label="randomize seed")
                with gr.Accordion("settings", open=False):
                    profile_name = gr.Textbox(label="profile name")
                    export_button = gr.Button("export settings", size="sm")
                    export_file = gr.File(label="profile", visible=False)
                    import_file = gr.File(label="import JSON profile", file_types=[".json"], type="filepath")
                    metadata_image = gr.Image(type="filepath", label="read settings from image")
                    read_metadata = gr.Button("read image metadata", size="sm")
                    profile_status = gr.Textbox(label="settings status", interactive=False)
                gen_budget = gr.Slider(0, MAX_GPU_SECONDS, value=0, step=10, label="GPU budget (0 = automatic)")
                button = gr.Button("generate", variant="primary", size="lg")
            with gr.Column(scale=1):
                gallery = gr.Gallery(label="output", columns=2, height=600)
                status = gr.Textbox(label="status", interactive=False)
                used_seed = gr.Number(label="used seed", interactive=False)

        def on_mode_change(value: str):
            editing = value == "edit"
            return gr.update(visible=editing), gr.update(visible=not editing), gr.update(visible=editing), gr.update(visible=not editing)

        mode.change(on_mode_change, inputs=[mode], outputs=[image_inputs, t2i_resolution, edit_controls, edit_prompt])

        generation_inputs = [mode, prompt, edit_prompt, primary, second, width, height, target_mp, grounding, ref_boost, ref_boost_a, steps, cfg, sampler, scheduler, seed, randomize, gen_budget]
        button.click(generate, inputs=generation_inputs, outputs=[gallery, status, used_seed])

        profile_inputs = [mode, prompt, edit_prompt, width, height, target_mp, grounding, ref_boost, ref_boost_a, steps, cfg, sampler, scheduler, seed, randomize, gen_budget, profile_name]

        def export_profile(*values):
            data = _profile_from_values(*values[:-1])
            name = re.sub(r"[^A-Za-z0-9_-]", "_", str(values[-1] or "krea2"))[:40]
            handle = tempfile.NamedTemporaryFile(prefix=f"{name}_", suffix=".json", mode="w", encoding="utf-8", delete=False)
            json.dump(data, handle, indent=2)
            handle.close()
            return handle.name, "settings exported"

        export_button.click(export_profile, inputs=profile_inputs, outputs=[export_file, profile_status]).then(lambda: gr.update(visible=True), outputs=[export_file])

        settings_outputs = [
            mode, prompt, edit_prompt, width, height, target_mp, grounding,
            ref_boost, ref_boost_a, steps, cfg, sampler, scheduler, seed,
            randomize, gen_budget, profile_status, image_inputs,
            t2i_resolution, edit_controls,
        ]
        settings_keys = [
            "mode", "prompt", "edit_prompt", "width", "height",
            "target_megapixels", "grounding_px", "ref_boost", "ref_boost_a",
            "steps", "cfg", "sampler_name", "scheduler", "effective_seed",
            "randomize_seed", "gen_budget",
        ]

        def settings_updates(data: dict[str, Any], warnings: list[str], error: str = ""):
            editing = data.get("mode") == "edit"
            updates = []
            for key in settings_keys:
                if key not in data:
                    updates.append(gr.update())
                elif key == "edit_prompt" and "mode" in data:
                    updates.append(gr.update(value=data[key], visible=editing))
                else:
                    updates.append(gr.update(value=data[key]))
            message = error or ("settings loaded" if not warnings else "; ".join(warnings[:6]))
            if len(warnings) > 6:
                message += f" (+{len(warnings) - 6} more)"
            updates.extend([
                message,
                gr.update(visible=editing) if "mode" in data else gr.update(),
                gr.update(visible=not editing) if "mode" in data else gr.update(),
                gr.update(visible=editing) if "mode" in data else gr.update(),
                gr.update(visible=editing) if "mode" in data else gr.update(),
            ])
            return tuple(updates)

        def import_profile(path: str | None):
            if not path:
                return settings_updates({}, [], "select a profile first")
            try:
                data, warnings = parse_settings_text(pathlib.Path(path).read_text(encoding="utf-8"))
            except Exception as exc:
                return settings_updates({}, [], f"could not read profile: {exc}")
            return settings_updates(data, warnings)

        import_file.change(import_profile, inputs=[import_file], outputs=settings_outputs)

        def import_metadata(path: str | None):
            if not path:
                return settings_updates({}, [], "upload an image first")
            data, warnings = extract_image_settings(path)
            return settings_updates(data, warnings)

        read_metadata.click(import_metadata, inputs=[metadata_image], outputs=settings_outputs)
    return demo


def _on_startup() -> None:
    if os.environ.get("KREA_SKIP_STARTUP") == "1":
        return
    try:
        _prepare_runtime()
    except Exception as exc:
        print(f"[startup] setup incomplete ({type(exc).__name__}: {exc}); generation will retry", flush=True)


_on_startup()
demo = create_ui()
demo.queue()

if __name__ == "__main__":
    demo.launch()
