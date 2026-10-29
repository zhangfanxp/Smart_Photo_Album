#!/usr/bin/env python3
"""Local natural-language photo search: structured filters + vector similarity.

Examples:
  python search.py -w "没有人的照片"        # JSON only, no model loading
  python search.py -w "夜晚海边没有人的照片"   # JSON + Embedding
  python search.py -w "只有一个小男孩"       # Count constraints + Embedding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from photo_core import (
    default_embedding_path, EMBED_VERSION, METADATA_DIR, VECTORS_DIR,
    TextEmbedder, derive_filters, read_json, text_hash, validate_model_path,
)
from query_parser import parse_query, meets_filters


def derive_query_constraints(q: str) -> list[tuple[str, str, object]]:
    """Compatibility helper for V1 callers and tests."""
    return list(parse_query(q).conditions)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地照片自然语言检索，只输出匹配图片的文件名称")
    p.add_argument("-w", "--words", required=True, help='如："夜晚海边没有人的照片"')
    p.add_argument("--top", type=int, default=10, help="最多返回多少张照片，默认 10")
    p.add_argument("--min-score", type=float, default=None, help="仅在使用向量检索时生效")
    p.add_argument("--no-filter", action="store_true", help="跳过全部结构化筛选，进行纯向量检索")
    p.add_argument("--show-score", action="store_true", help="同时显示余弦相似度（纯 JSON 查询无分数）")
    p.add_argument("--full-path", action="store_true", help="显示绝对路径")
    p.add_argument("--explain", action="store_true", help="在标准错误输出解释查询拆分，不影响图片名称输出")
    p.add_argument("--embedding-model", type=Path, default=default_embedding_path(), help="本地向量模型目录")
    return p.parse_args()


def load_candidates(model_path: Path | None, constraints: list, require_vectors: bool) -> list[tuple[dict, np.ndarray | None]]:
    records = []
    for path in sorted(METADATA_DIR.glob("*.json")):
        record = read_json(path)
        if not record or not isinstance(record.get("analysis"), dict):
            continue
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
    query_text = args.words if args.no_filter else plan.semantic_query
    # Any query with residual semantic content uses the embedding model. A pure
    # hard-filter query (e.g. '没有人的照片') only touches the local JSON metadata.
    vector_search = bool(query_text.strip()) or not constraints

    if args.explain:
        print(f"结构化条件：{constraints}", file=sys.stderr)
        print(f"语义部分：{query_text!r}", file=sys.stderr)
        print(f"查询方式：{'JSON 筛选 + 向量排序' if vector_search else '仅 JSON 精确筛选'}", file=sys.stderr)
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

    records = load_candidates(model_path, constraints, vector_search)
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
    indices = np.argsort(-scores, kind="stable")
    printed = 0
    for idx in indices:
        score = float(scores[idx])
        if args.min_score is not None and score < args.min_score:
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
