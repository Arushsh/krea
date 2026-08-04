"""Unit tests for Krea profile and image metadata helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from settings_utils import (
    APP_ID,
    build_settings,
    extract_image_settings,
    parse_settings_text,
    write_png_metadata,
)


class SettingsTests(unittest.TestCase):
    def settings(self):
        return build_settings(
            mode="edit",
            prompt="",
            edit_prompt="change the coat to blue",
            width=1024,
            height=768,
            target_megapixels=1.4,
            grounding_px=768,
            ref_boost=1.0,
            ref_boost_a=1.0,
            steps=8,
            cfg=1.0,
            sampler_name="euler",
            scheduler="beta",
            seed=2,
            randomize_seed=False,
            gen_budget=0,
            effective_seed=2,
        )

    def test_png_metadata_round_trip(self):
        settings = self.settings()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            destination = Path(directory) / "output.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
            write_png_metadata(source, destination, settings)
            loaded, warnings = extract_image_settings(destination)
        self.assertEqual(loaded["app"], APP_ID)
        self.assertEqual(loaded["edit_prompt"], "change the coat to blue")
        self.assertEqual(loaded["ref_boost"], 1.0)
        self.assertEqual(warnings, [])

    def test_json_profile_round_trip(self):
        parsed, warnings = parse_settings_text(__import__("json").dumps(self.settings()))
        self.assertEqual(warnings, [])
        self.assertEqual(parsed["mode"], "edit")
        self.assertEqual(parsed["grounding_px"], 768)

    def test_a1111_parameter_text(self):
        parsed, warnings = parse_settings_text(
            "a portrait\nSteps: 8, CFG scale: 1, Sampler: euler, Seed: 7, Size: 512x768"
        )
        self.assertEqual(warnings, [])
        self.assertEqual(parsed["prompt"], "a portrait")
        self.assertEqual(parsed["width"], 512)
        self.assertEqual(parsed["effective_seed"], 7)


if __name__ == "__main__":
    unittest.main()
