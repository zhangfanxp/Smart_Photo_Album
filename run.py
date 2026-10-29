#!/usr/bin/env python3
"""Batch photo analysis + text vector indexing. Example: python run.py ./images"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from tqdm import tqdm

from photo_core import (
    DEFAULT_EMBED, default_embedding_path, DEFAULT_VLM, EMBED_VERSION, ERROR_DIR, IMAGE_EXTENSIONS,
    METADATA_DIR, PROJECT_DIR, VECTORS_DIR, ImageAnalyzer, TextEmbedder,
    atomic_json, atomic_vector, derive_filters, image_id, make_dirs,
    prepare_image, read_json, stat_signature, text_hash, timestamp,
    validate_model_path,
)
from prompts import ANALYSIS_PROMPT, PROMPT_VERSION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="识别指定文件夹的图片，保存 JSON 和 768 维向量 (.npy)")
    p.add_argument("images", nargs="?", default="./images", help="图片目录，默认 ./images（支持子目录）")
    p.add_argument("--vlm-model", type=Path, default=DEFAULT_VLM, help="本地 Qwen3-VL MLX 目录")
    p.add_argument("--embedding-model", type=Path, default=default_embedding_path(), help="本地 EmbeddingGemma 2 MLX 目录")
    p.add_argument("--max-side", type=int, default=1600, help="分析图片最长边，默认 1600 像素")
    p.add_argument("--max-tokens", type=int, default=2200, help="Qwen3-VL 最大输出 token 数")
    p.add_argument("--force", action="store_true", help="不使用缓存，重新识别并向量化")
    return p.parse_args()


def scan_images(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower()
    )


def meta_path(img: Path) -> Path:
    return METADATA_DIR / f"{image_id(img)}.json"


def vec_path(img: Path) -> Path:
    return VECTORS_DIR / f"{image_id(img)}.npy"


def vecmeta_path(img: Path) -> Path:
    return VECTORS_DIR / f"{image_id(img)}.json"


def is_analysis_current(img: Path, old: dict | None, model: Path, stat: dict) -> bool:
    return bool(
        old and old.get("file_signature") == stat
        and old.get("prompt_version") == PROMPT_VERSION
        and old.get("vlm_model") == str(model)
        and isinstance((old.get("analysis") or {}).get("search_text"), str)
        and len(old["analysis"]["search_text"]) > 7
    )


def is_vector_current(img: Path, record: dict, model: Path) -> bool:
    vmeta = read_json(vecmeta_path(img))
    return bool(
        vec_path(img).is_file() and vmeta
        and vmeta.get("embedding_model") == str(model)
        and vmeta.get("embedding_version") == EMBED_VERSION
        and vmeta.get("search_text_hash") == text_hash(record["analysis"]["search_text"])
        and vmeta.get("dimension") == 768
    )


def main() -> int:
    args = parse_args()
    if args.max_side < 256:
        print("--max-side 必须 >= 256", file=sys.stderr)
        return 2
    root = Path(args.images).expanduser().resolve()
    if not root.is_dir():
        print(f"图片目录不存在：{root}", file=sys.stderr)
        return 2
    files = scan_images(root)
    if not files:
        print(f"没有找到图片：{root}\n支持 JPG、PNG、WEBP、HEIC、TIFF 等格式。")
        return 0
    try:
        vlm_model = validate_model_path(args.vlm_model, "Qwen3-VL")
        embed_model = validate_model_path(args.embedding_model, "EmbeddingGemma 2")
    except Exception as exc:
        print(f"模型检查失败：{exc}", file=sys.stderr)
        return 2

    make_dirs()
    print(f"图片目录：{root}")
    print(f"发现图片：{len(files)} 张")
    print(f"Qwen3-VL：{vlm_model}")
    print(f"Embedding：{embed_model}")

    stats = {}
    pending = []
    for img in files:
        try:
            signature = stat_signature(img)
            stats[img] = signature
            old = read_json(meta_path(img))
            if args.force or not is_analysis_current(img, old, vlm_model, signature):
                pending.append(img)
        except OSError as exc:
            print(f"无法读取文件 {img}: {exc}", file=sys.stderr)
    failed = {}
    analyzed = 0
    if pending:
        print(f"\n第一阶段：Qwen3-VL 图片识别（待处理 {len(pending)} 张）")
        try:
            analyzer = ImageAnalyzer(vlm_model)
        except Exception as exc:
            print(f"Qwen3-VL 加载失败：{exc}", file=sys.stderr)
            return 2
        scratch = PROJECT_DIR / ".cache" / "prepared"
        scratch.mkdir(parents=True, exist_ok=True)
        for img in tqdm(pending, desc="识别图片", unit="张", dynamic_ncols=True):
            prepared = scratch / f"{image_id(img)}.jpg"
            try:
                prepare_image(img, prepared, args.max_side)
                analysis, raw = analyzer.analyze(prepared, ANALYSIS_PROMPT, args.max_tokens)
                entry = {
                    "id": image_id(img),
                    "image_path": str(img),
                    "relative_path": img.relative_to(root).as_posix(),
                    "image_root": str(root),
                    "file_signature": stats[img],
                    "prompt_version": PROMPT_VERSION,
                    "vlm_model": str(vlm_model),
                    "indexed_at": timestamp(),
                    "analysis": analysis,
                    "filters": derive_filters(analysis),
                }
                atomic_json(meta_path(img), entry)
                analyzed += 1
                (ERROR_DIR / f"{image_id(img)}.txt").unlink(missing_ok=True)
            except Exception as exc:
                failed[img] = str(exc)
                raw = getattr(analyzer, "last_raw", "")
                (ERROR_DIR / f"{image_id(img)}.txt").write_text(
                    f"文件: {img}\n错误: {exc}\n\n模型输出:\n{raw}", encoding="utf-8"
                )
                tqdm.write(f"[识别失败] {img.name}: {exc}")
            finally:
                prepared.unlink(missing_ok=True)
        del analyzer
        gc.collect()
        try:
            import mlx.core as mx
            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
            elif hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            pass
    else:
        print("\n第一阶段：图片识别缓存有效，跳过。")

    # Second pass: sequential loading keeps memory lower on 16GB Apple Silicon.
    to_embed = []
    for img in files:
        entry = read_json(meta_path(img))
        if (img not in failed and entry
                and is_analysis_current(img, entry, vlm_model, stats.get(img))
                and (args.force or not is_vector_current(img, entry, embed_model))):
            to_embed.append((img, entry))
    embedded = 0
    if to_embed:
        print(f"\n第二阶段：EmbeddingGemma 2 文本向量化（待处理 {len(to_embed)} 张）")
        try:
            encoder = TextEmbedder(embed_model)
        except Exception as exc:
            print(f"Embedding 模型加载失败：{exc}", file=sys.stderr)
            print("已生成的 metadata/*.json 会被保留，修复环境后重新执行本命令即可续跑。")
            return 2
        for img, entry in tqdm(to_embed, desc="生成向量", unit="张", dynamic_ncols=True):
            try:
                desc = entry["analysis"]["search_text"]
                vector = encoder.encode(desc, query=False)
                atomic_vector(vec_path(img), vector)
                atomic_json(vecmeta_path(img), {
                    "image_id": image_id(img),
                    "embedding_model": str(embed_model),
                    "embedding_version": EMBED_VERSION,
                    "search_text_hash": text_hash(desc),
                    "dimension": int(vector.size),
                    "created_at": timestamp(),
                })
                embedded += 1
            except Exception as exc:
                failed[img] = f"向量化失败：{exc}"
                tqdm.write(f"[向量化失败] {img.name}: {exc}")
        del encoder
        gc.collect()
    else:
        print("第二阶段：向量缓存有效，跳过。")

    valid_count = 0
    for img in files:
        entry = read_json(meta_path(img))
        if entry and is_analysis_current(img, entry, vlm_model, stats.get(img)) and is_vector_current(img, entry, embed_model):
            valid_count += 1

    print("\n" + "=" * 48)
    print(f"扫描图片：{len(files)} 张 | 本次识别：{analyzed} 张 | 本次向量化：{embedded} 张")
    print(f"索引完整：{valid_count}/{len(files)} 张")
    print(f"图片描述位置：{METADATA_DIR}")
    print(f"向量文件位置：{VECTORS_DIR}")
    if valid_count == len(files):
        print("✅ 图片识别且向量化完成！")
        print('现在可以运行：python search.py -w "一个小男孩"')
        return 0
    print(f"⚠️ 部分图片尚未完成，共 {len(files)-valid_count} 张。可再次运行相同命令续跑。")
    print(f"识别错误详情：{ERROR_DIR}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。成功保存的 JSON / 向量会保留，下次自动续跑。", file=sys.stderr)
        sys.exit(130)
