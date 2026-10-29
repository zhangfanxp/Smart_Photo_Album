"""V2 query behavior tests, no MLX models or network required."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import photo_core as core
import query_parser as qp
import search


class QueryParserTest(unittest.TestCase):
    def test_negation_is_not_positive_people(self):
        plan = qp.parse_query("没有人的照片")
        self.assertEqual(plan.semantic_query, "")
        self.assertIn(("people_count", "eq", 0), plan.conditions)
        self.assertIn(("people_count_exact", "eq", True), plan.conditions)
        self.assertFalse(any(c[0] == "people_count" and c[1] == "ge" for c in plan.conditions))

    def test_mixed_query(self):
        plan = qp.parse_query("夜晚海边没有人的照片")
        self.assertEqual(plan.semantic_query, "海边")
        self.assertIn(("time_of_day", "eq", "night"), plan.conditions)
        self.assertIn(("people_count", "eq", 0), plan.conditions)

    def test_zero_children_is_not_zero_people(self):
        plan = qp.parse_query("没有小孩的照片")
        self.assertIn(("child_count", "eq", 0), plan.conditions)
        self.assertIn(("person_age_complete", "eq", True), plan.conditions)
        self.assertNotIn(("people_count", "eq", 0), plan.conditions)

    def test_existing_queries(self):
        plan = qp.parse_query("只有一个小男孩")
        self.assertIn(("people_count", "eq", 1), plan.conditions)
        self.assertIn(("male_child_count", "ge", 1), plan.conditions)
        self.assertEqual(qp.parse_query("有三个中年男人").semantic_query, "中年男人")
        plan = qp.parse_query("一个女性和两个小孩")
        self.assertIn(("adult_female_count", "ge", 1), plan.conditions)
        self.assertIn(("child_count", "ge", 2), plan.conditions)

    def test_unknown_age_never_passes_no_children_filter(self):
        analysis = {
            "people": {"count": 1, "count_exact": True, "persons": [{"age_group": "unknown", "gender_presentation": "unknown"}]},
            "animals": {"count": 0, "count_exact": True, "items": []}, "scene": {},
        }
        flags = core.derive_filters(analysis)
        self.assertFalse(flags["person_age_complete"])
        self.assertFalse(qp.meets_filters(flags, qp.parse_query("没有小孩").conditions))

    def test_zero_people_search_works_without_embedding_model(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            meta = home / "metadata"
            meta.mkdir()
            images = home / "images"
            images.mkdir()
            for name, count, exact, persons in [
                ("empty.jpg", 0, True, []),
                ("person.jpg", 1, True, [{"age_group": "adult", "gender_presentation": "female"}]),
                ("uncertain.jpg", 0, False, []),
            ]:
                im = images / name
                im.touch()
                entry = {
                    "image_path": str(im), "relative_path": name,
                    "analysis": {
                        "search_text": "一张测试照片的描述信息。",
                        "people": {"count": count, "count_exact": exact, "persons": persons},
                        "animals": {"count": 0, "count_exact": True, "items": []},
                        "scene": {},
                    },
                }
                (meta / f"{name}.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            out = io.StringIO()
            with patch.object(search, "METADATA_DIR", meta), \
                 patch.object(search, "VECTORS_DIR", home / "not-built-vectors"), \
                 patch.object(search, "TextEmbedder", side_effect=AssertionError("should not load embeddings")), \
                 patch.object(search, "validate_model_path", side_effect=AssertionError("should not validate model")), \
                 patch.object(sys, "argv", ["search.py", "-w", "没有人的照片"]), \
                 contextlib.redirect_stdout(out):
                self.assertEqual(search.main(), 0)
            self.assertEqual(out.getvalue().strip(), "empty.jpg")


if __name__ == "__main__":
    unittest.main()
