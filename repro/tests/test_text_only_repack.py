import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "repack_official_litertlm.py"
SPEC = importlib.util.spec_from_file_location("repack_official_litertlm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TextOnlySectionsTest(unittest.TestCase):
    def test_removes_only_audio_vision_sections_and_keeps_mtp(self):
        names = [
            "embedder", "per_layer_embedder", "audio_encoder_hw", "audio_adapter",
            "end_of_audio", "vision_encoder", "vision_adapter", "end_of_vision",
            "prefill_decode", "mtp_drafter",
        ]
        source = "[system_metadata]\nentries = []\n\n" + "".join(
            f'[[section]]\nmodel_type = "{name}"\nsection_type = "TFLiteModel"\n'
            f'data_path = "Section_{name}.tflite"\n\n'
            for name in names
        )
        result, removed = MODULE.remove_multimodal_sections(source)
        self.assertEqual(set(removed), MODULE.TEXT_ONLY_MODEL_TYPES)
        for name in ("embedder", "per_layer_embedder", "prefill_decode", "mtp_drafter"):
            self.assertIn(f'model_type = "{name}"', result)
        for name in MODULE.TEXT_ONLY_MODEL_TYPES:
            self.assertNotIn(f'model_type = "{name}"', result)

    def test_fails_closed_when_multimodal_section_is_missing(self):
        source = '[[section]]\nmodel_type = "mtp_drafter"\n'
        with self.assertRaisesRegex(ValueError, "expected all six"):
            MODULE.remove_multimodal_sections(source)


if __name__ == "__main__":
    unittest.main()
