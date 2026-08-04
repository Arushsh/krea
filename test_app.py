"""Pure runtime tests; no ComfyUI clone or model download is required."""

from __future__ import annotations

import os
import sys
import unittest

os.environ["KREA_SKIP_STARTUP"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import app  # noqa: E402


class WorkflowTests(unittest.TestCase):
    def test_text_to_image_loader_and_prompt_injection(self):
        workflow = app._t2i_workflow()
        app._inject_t2i(
            workflow,
            prompt="a blue glass sculpture",
            width=1024,
            height=768,
            steps=8,
            cfg=1.0,
            sampler="euler",
            scheduler="beta",
            seed=42,
        )
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], "krea2_turbo_fp8_scaled.safetensors")
        self.assertEqual(workflow["2"]["inputs"]["clip_name"], "qwen3vl_4b_fp8_scaled.safetensors")
        self.assertEqual(workflow["5"]["inputs"]["text"], "a blue glass sculpture")
        self.assertEqual(workflow["7"]["inputs"]["height"], 768)
        self.assertEqual(workflow["8"]["inputs"]["seed"], 42)

    def test_edit_workflow_has_mandatory_adapter_and_optional_second_reference(self):
        workflow = app._edit_workflow(True)
        self.assertEqual(workflow["6"]["class_type"], "LoraLoaderModelOnly")
        self.assertTrue(workflow["6"]["inputs"]["lora_name"].endswith(app.IDENTITY_FILE))
        self.assertIn("source_latent_b", workflow["9"]["inputs"])
        self.assertIn("image_b", workflow["10"]["inputs"])

    def test_edit_workflow_without_second_reference_omits_b_inputs(self):
        workflow = app._edit_workflow(False)
        self.assertNotIn("2", workflow)
        self.assertNotIn("source_latent_b", workflow["9"]["inputs"])
        self.assertNotIn("source_image_b", workflow["9"]["inputs"])
        self.assertNotIn("image_b", workflow["10"]["inputs"])

    def test_mode_validation(self):
        with self.assertRaises(ValueError):
            app._validate_request("text2image", "", "", None)
        with self.assertRaises(ValueError):
            app._validate_request("edit", "", "change the color", None)
        with self.assertRaises(ValueError):
            app._validate_request("edit", "", "", "source.png")
        app._validate_request("edit", "", "change the color", "source.png")

    def test_gpu_duration_accepts_spaces_positional_callback(self):
        values = [
            "text2image", "prompt", "", None, None,
            1024, 1024, 1.4, 768, 1.0, 1.0,
            8, 1.0, "euler", "beta", 2, False, 0, None,
        ]
        duration = app.get_gpu_duration(*values)
        self.assertGreaterEqual(duration, app.MIN_GPU_SECONDS)

    def test_gpu_duration_accepts_keyword_callback(self):
        duration = app.get_gpu_duration(steps=8, width=1024, height=1024, gen_budget=120)
        self.assertEqual(duration, 120)


if __name__ == "__main__":
    unittest.main()
