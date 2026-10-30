#!/usr/bin/env python3
"""Local natural-language photo search: structured filters + vector similarity.

Examples:
  python search.py -w "没有人的照片"        # JSON only, no model loading
  python search.py -w "夜晚海边没有人的照片"   # JSON + Embedding
  python search.py -w "只有一个小男孩"       # Count constraints + Embedding
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import re
import sys
from pathlib import Path

import numpy as np

from photo_core import (
    default_embedding_path, EMBED_VERSION, METADATA_DIR, VECTORS_DIR,
    PROJECT_DIR, TextEmbedder, derive_filters, read_json, text_hash, validate_model_path,
)
from query_parser import parse_query, meets_filters

ANNOTATION_OVERRIDES_PATH = PROJECT_DIR / "annotation_overrides.json"
SOFT_PERSON_COUNT_FIELDS = {
    "male_child_count", "female_child_count", "child_count",
    "adult_female_count", "middle_aged_male_count",
}
SOFT_MATCH_BONUS = 0.08
MANUAL_LABEL_BONUS = 0.15
# Default cosine-similarity cutoff for every vector search. This is not a probability.
DEFAULT_MIN_SCORE = 0.69


def derive_query_constraints(q: str) -> list[tuple[str, str, object]]:
    """Compatibility helper for V1 callers and tests."""
    return list(parse_query(q).conditions)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地照片自然语言检索，只输出匹配图片的文件名称")
    p.add_argument("-w", "--words", required=True, help='如："夜晚海边没有人的照片"')
    p.add_argument("--top", type=int, default=10, help="最多返回多少张照片，默认 10")
    p.add_argument("--min-score", type=float, default=None, help="覆盖默认最低余弦相似度 0.69，可调低以扩大召回")
    p.add_argument("--no-filter", action="store_true", help="跳过全部结构化筛选，进行纯向量检索")
    p.add_argument("--show-score", action="store_true", help="显示余弦相似度（排序可能结合人物属性和人工标注）")
    p.add_argument("--full-path", action="store_true", help="显示绝对路径")
    p.add_argument("--explain", action="store_true", help="在标准错误输出解释查询拆分，不影响图片名称输出")
    p.add_argument("--embedding-model", type=Path, default=default_embedding_path(), help="本地向量模型目录")
    return p.parse_args()


def split_conditions(conditions: list) -> tuple[list, list]:
    """Keep total counts and explicit exclusions strict; rank positive demographics softly."""
    hard, soft = [], []
    for condition in conditions:
        key, op, expected = condition
        if key in SOFT_PERSON_COUNT_FIELDS and op == "ge" and expected > 0:
            soft.append(condition)
        else:
            hard.append(condition)
    return hard, soft


def load_annotation_overrides() -> dict:
    """Load human corrections kept separately from VLM-generated metadata."""
    data = read_json(ANNOTATION_OVERRIDES_PATH) or {}
    images = data.get("images", {})
    return images if isinstance(images, dict) else {}


def _with_annotation_override(record: dict, override: dict | None) -> dict:
    if not isinstance(override, dict):
        return record
    annotated = dict(record)
    analysis = deepcopy(record["analysis"])
    people = analysis.get("people")
    persons = people.get("persons", []) if isinstance(people, dict) else []
    person_overrides = override.get("persons", {})
    if isinstance(person_overrides, dict):
        for person in persons:
            if not isinstance(person, dict):
                continue
            changes = person_overrides.get(str(person.get("id", "")))
            if not isinstance(changes, dict):
                continue
            for field, allowed in (
                ("gender_presentation", {"male", "female", "unknown"}),
                ("age_group", {"infant", "toddler", "child", "teenager", "young_adult", "middle_aged", "elderly", "unknown"}),
            ):
                if changes.get(field) in allowed:
                    person[field] = changes[field]
    scene_overrides = override.get("scene", {})
    if isinstance(scene_overrides, dict):
        scene_fields = analysis.get("scene")
        if not isinstance(scene_fields, dict):
            scene_fields = {}
            analysis["scene"] = scene_fields
        for field in ("sky_blue", "red_flag_visible", "red_handprint_visible"):
            value = scene_overrides.get(field)
            if field in scene_overrides and (type(value) is bool or value is None):
                scene_fields[field] = value
    annotated["analysis"] = analysis
    labels = override.get("search_labels", [])
    if isinstance(labels, list):
        annotated["_manual_search_labels"] = list(dict.fromkeys(
            label.strip() for label in labels if isinstance(label, str) and label.strip()
        ))
    return annotated


def load_candidates(
    model_path: Path | None,
    constraints: list,
    require_vectors: bool,
    annotation_overrides: dict | None = None,
) -> list[tuple[dict, np.ndarray | None]]:
    records = []
    for path in sorted(METADATA_DIR.glob("*.json")):
        record = read_json(path)
        if not record or not isinstance(record.get("analysis"), dict):
            continue
        record = _with_annotation_override(record, (annotation_overrides or {}).get(path.stem))
        img = Path(str(record.get("image_path", "")))
        if not img.is_file():
            continue
        # V2 computes the extended proof flags from old JSON on the fly.
        # This makes the user's original 21 photos searchable without run --force.
        fields = derive_filters(record["analysis"])
        if constraints and not meets_filters(fields, constraints):
            continue
        if not require_vectors:
            records.append((record, None))
            continue
        vectorfile = VECTORS_DIR / f"{path.stem}.npy"
        vmeta = read_json(VECTORS_DIR / f"{path.stem}.json")
        if not vectorfile.is_file() or not vmeta:
            continue
        description = record["analysis"].get("search_text", "")
        if (vmeta.get("embedding_model") != str(model_path)
                or vmeta.get("embedding_version") != EMBED_VERSION
                or vmeta.get("search_text_hash") != text_hash(description)):
            continue
        try:
            vector = np.load(vectorfile, allow_pickle=False)
        except (OSError, ValueError):
            continue
        if vector.shape != (768,) or not np.isfinite(vector).all():
            continue
        records.append((record, vector.astype(np.float32)))
    return records


def display_name(record: dict, full_path: bool) -> str:
    return str(record["image_path"]) if full_path else str(
        record.get("relative_path") or Path(record["image_path"]).name
    )


def _soft_condition_score(fields: dict, conditions: list) -> float:
    if not conditions:
        return 0.0
    matched = sum(meets_filters(fields, [condition]) for condition in conditions)
    return matched / len(conditions)


def _manual_label_matches(query: str, labels: list[str]) -> bool:
    normalized_query = re.sub(r"\s+", "", query)
    return bool(normalized_query and any(
        label in normalized_query or normalized_query in label for label in labels
    ))


def main() -> int:
    args = parse_args()
    if args.top < 1:
        print("--top 必须大于 0", file=sys.stderr)
        return 2
    if not args.words.strip():
        print("搜索条件不能为空", file=sys.stderr)
        return 2
    if not METADATA_DIR.exists():
        print("找不到 metadata/，请先运行：python run.py ./images", file=sys.stderr)
        return 2

    plan = parse_query(args.words)
    constraints = [] if args.no_filter else list(plan.conditions)
    hard_constraints, soft_constraints = split_conditions(constraints)
    query_text = args.words if args.no_filter else plan.semantic_query
    # A pure hard-filter query has no vector score. All other vector searches
    # use the default cutoff unless the user explicitly overrides it.
    vector_search = bool(query_text.strip()) or not hard_constraints
    if not vector_search:
        min_score = None
        min_score_source = "纯 JSON 查询不适用"
    elif args.min_score is not None:
        min_score = args.min_score
        min_score_source = "命令行"
    else:
        min_score = DEFAULT_MIN_SCORE
        min_score_source = "系统默认值"
    # Any query with residual semantic content uses the embedding model. A pure
    # hard-filter query (e.g. '没有人的照片') only touches the local JSON metadata.

    if args.explain:
        print(f"硬筛选条件：{hard_constraints}", file=sys.stderr)
        print(f"软排序条件：{soft_constraints}", file=sys.stderr)
        print(f"语义部分：{query_text!r}", file=sys.stderr)
        print(f"查询方式：{'JSON 筛选 + 向量排序' if vector_search else '仅 JSON 精确筛选'}", file=sys.stderr)
        print(f"最低相似度：{min_score if min_score is not None else '无'}（{min_score_source}）", file=sys.stderr)
        if not args.no_filter:
            print(f"人工标注覆盖：{ANNOTATION_OVERRIDES_PATH}", file=sys.stderr)
    if plan.unsupported_logic and not args.no_filter:
        print("提示：当前版本不支持‘或/或者’的完整逻辑组合；请改成两次查询。", file=sys.stderr)
    if not vector_search and args.min_score is not None:
        print("提示：纯 JSON 筛选无需相似度，--min-score 在本次查询中被忽略。", file=sys.stderr)

    model_path = None
    if vector_search:
        try:
            model_path = validate_model_path(args.embedding_model, "EmbeddingGemma 2")
        except Exception as exc:
            print(f"向量模型检查失败：{exc}", file=sys.stderr)
            return 2

    annotation_overrides = {} if args.no_filter else load_annotation_overrides()
    records = load_candidates(model_path, hard_constraints, vector_search, annotation_overrides)
    if not records:
        print("未找到符合条件的照片。", file=sys.stderr)
        return 0

    if not vector_search:
        # For a purely structured query there is no meaningful similarity
        # score: order matches deterministically by filename.
        for record, _ in sorted(records, key=lambda item: display_name(item[0], False).lower())[:args.top]:
            print(display_name(record, args.full_path))
        return 0

    try:
        encoder = TextEmbedder(model_path)
        query_vector = encoder.encode(query_text, query=True)
    except Exception as exc:
        print(f"搜索模型加载或编码失败：{exc}", file=sys.stderr)
        return 2

    vectors = np.stack([vec for _, vec in records])
    scores = vectors @ query_vector
    rank_scores = scores.copy()
    if soft_constraints:
        for i, (record, _) in enumerate(records):
            fields = derive_filters(record["analysis"])
            rank_scores[i] += SOFT_MATCH_BONUS * _soft_condition_score(fields, soft_constraints)
    if query_text.strip():
        for i, (record, _) in enumerate(records):
            labels = record.get("_manual_search_labels", [])
            if _manual_label_matches(query_text, labels):
                rank_scores[i] += MANUAL_LABEL_BONUS
    indices = np.argsort(-rank_scores, kind="stable")
    printed = 0
    for idx in indices:
        score = float(scores[idx])
        if min_score is not None and score < min_score:
            continue
        entry = records[int(idx)][0]
        name = display_name(entry, args.full_path)
        print(f"{name}\t{score:.4f}" if args.show_score else name)
        printed += 1
        if printed == args.top:
            break
    if not printed:
        print("没有达到指定相似度阈值的照片。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消搜索。", file=sys.stderr)
        sys.exit(130)
