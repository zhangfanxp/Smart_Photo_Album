import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import photo_core as core
import run
import search


class DummyVLM:
    def __init__(self, path):
        self.last_raw = ''

    def analyze(self, prepared_image, prompt, max_tokens):
        analysis = {
            'image_type': 'photo', 'categories': ['people', 'nature'],
            'summary': '一名男孩在公园里',
            'search_text': '公园中只有一个小男孩，穿蓝色上衣，站在草地上。',
            'people': {'count': 1, 'count_exact': True, 'persons': [{'id': 'p1', 'gender_presentation': 'male', 'age_group': 'child'}]},
            'animals': {'count': 0, 'count_exact': True, 'items': []},
            'scene': {'location_type': 'park', 'time_of_day': 'daytime'},
        }
        return core.validate_analysis(analysis), json.dumps(analysis)


class DummyEncoder:
    def __init__(self, path):
        pass

    def encode(self, text, *, query=False):
        return np.full(768, 1.0 / np.sqrt(768), dtype=np.float32)


class PipelineTest(unittest.TestCase):
    def test_parsing_and_constraints(self):
        got = core.read_model_json('Here is the json:\n```json\n{"a": 1}\n```')
        self.assertEqual(got, {'a': 1})
        q = search.derive_query_constraints('只有一个小男孩')
        self.assertIn(('people_count', 'eq', 1), q)
        self.assertIn(('male_child_count', 'ge', 1), q)
        self.assertFalse(search.meets_filters({'people_count': 2, 'people_count_exact': True, 'male_child_count': 1}, q))
        self.assertTrue(search.meets_filters({'people_count': 1, 'people_count_exact': True, 'male_child_count': 1}, q))
        q = search.derive_query_constraints('一个女性和两个小孩')
        self.assertIn(('adult_female_count', 'ge', 1), q)
        self.assertIn(('child_count', 'ge', 2), q)

    def test_end_to_end_fake_models(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images = root / 'images'
            images.mkdir()
            image_path = images / 'a.jpg'
            Image.new('RGB', (64, 64), 'white').save(image_path)
            meta, vec, errors = root/'metadata', root/'vectors', root/'errors'
            meta.mkdir(); vec.mkdir(); errors.mkdir()
            fakevlm, fakeembed = root/'model'/'vlm', root/'model'/'emb'
            fakevlm.mkdir(parents=True); fakeembed.mkdir(parents=True)
            setup = [
                patch.object(run, 'METADATA_DIR', meta), patch.object(run, 'VECTORS_DIR', vec),
                patch.object(run, 'ERROR_DIR', errors), patch.object(run, 'PROJECT_DIR', root),
                patch.object(run, 'ImageAnalyzer', DummyVLM), patch.object(run, 'TextEmbedder', DummyEncoder),
                patch.object(run, 'validate_model_path', lambda p, why: Path(p).resolve()),
                patch.object(core, 'METADATA_DIR', meta), patch.object(core, 'VECTORS_DIR', vec),
                patch.object(search, 'METADATA_DIR', meta), patch.object(search, 'VECTORS_DIR', vec),
                patch.object(search, 'TextEmbedder', DummyEncoder),
                patch.object(search, 'validate_model_path', lambda p, why: Path(p).resolve()),
            ]
            with contextlib.ExitStack() as stack:
                for context in setup:
                    stack.enter_context(context)
                with patch.object(sys, 'argv', ['run.py', str(images), '--vlm-model', str(fakevlm), '--embedding-model', str(fakeembed)]):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(run.main(), 0)
                    self.assertIn('图片识别且向量化完成', output.getvalue())
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(run.main(), 0)  # resumed
                self.assertEqual(len(list(meta.glob('*.json'))), 1)
                self.assertEqual(len(list(vec.glob('*.npy'))), 1)
                result = io.StringIO()
                with patch.object(sys, 'argv', ['search.py', '-w', '只有一个小男孩', '--embedding-model', str(fakeembed)]):
                    with contextlib.redirect_stdout(result):
                        self.assertEqual(search.main(), 0)
                self.assertEqual(result.getvalue().strip(), 'a.jpg')

if __name__ == '__main__':
    unittest.main()
