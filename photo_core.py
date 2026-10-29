"""Shared local MLX and file utilities for the photo indexing proof of concept."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VLM = PROJECT_DIR / "model" / "Qwen3-VL-4B-Instruct-4bit"
DEFAULT_EMBED = PROJECT_DIR / "model" / "embeddinggemma-2-4bit"
DEFAULT_EMBED_ORIGINAL = PROJECT_DIR / "model" / "embeddinggemma-2"


def default_embedding_path() -> Path:
    """Prefer MLX; use downloaded ModelScope original if it is the only copy."""
    if DEFAULT_EMBED.exists():
        return DEFAULT_EMBED
    if DEFAULT_EMBED_ORIGINAL.exists():
        return DEFAULT_EMBED_ORIGINAL
    return DEFAULT_EMBED
METADATA_DIR = PROJECT_DIR / "metadata"
VECTORS_DIR = PROJECT_DIR / "vectors"
ERROR_DIR = PROJECT_DIR / "errors"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
EMBED_VERSION = "embeddinggemma2-text-retrieval-v1"
VECTOR_DIMENSION = 768


def make_dirs() -> None:
    for d in (METADATA_DIR, VECTORS_DIR, ERROR_DIR):
        d.mkdir(parents=True, exist_ok=True)


def validate_model_path(path: Path, purpose: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{purpose} 模型目录不存在：{path}\n请检查 model/ 下的实际文件夹名称，或通过命令行参数指定。")
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        raise RuntimeError(f"{purpose} 目录缺少 config.json 或 *.safetensors：{path}\n请下载完整的模型文件。")
    if purpose == "EmbeddingGemma 2":
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if "embedding" not in str(cfg.get("model_type", "")).lower():
            raise RuntimeError(f"Embedding 模型配置不符合预期：{path} (model_type={cfg.get('model_type')!r})")
        # Both the MLX 4bit and the original Transformers checkpoints are valid.
        # They are loaded using different backends below.
    return path


def image_id(image: Path) -> str:
    return hashlib.sha256(str(image.resolve()).encode("utf-8")).hexdigest()[:24]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_vector(path: Path, vec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, vec.astype(np.float32, copy=False), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stat_signature(image: Path) -> dict[str, int]:
    stat = image.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_model_json(raw: str) -> dict[str, Any]:
    """Decode JSON even if a small model wraps it in markdown or extra text."""
    raw = raw.strip()
    decoder = json.JSONDecoder()
    for i, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("模型未返回有效 JSON")


def _integer(value: Any, default: int = 0) -> int:
    try:
        result = int(value)
        return max(0, result)
    except (ValueError, TypeError):
        return default


def _is_true(value: Any) -> bool:
    return value is True  # never interpret unknown/strings as exact


def validate_analysis(a: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(a.get("search_text"), str) or len(a["search_text"].strip()) < 8:
        raise ValueError("模型输出缺少有效的 search_text")
    a["search_text"] = a["search_text"].strip()
    a["summary"] = str(a.get("summary") or "").strip()
    a["categories"] = [s for s in a.get("categories", []) if isinstance(s, str)] if isinstance(a.get("categories"), list) else []
    a["tags"] = [s for s in a.get("tags", []) if isinstance(s, str)] if isinstance(a.get("tags"), list) else []
    if not isinstance(a.get("scene"), dict):
        a["scene"] = {}
    for name, item_key in (("people", "persons"), ("animals", "items")):
        section = a.get(name) or {}
        if not isinstance(section, dict):
            section = {}
        items = section.get(item_key, [])
        if not isinstance(items, list):
            items = []
        section[item_key] = [x for x in items if isinstance(x, dict)]
        # If model forgot count, derive from visible entities but NOT exactness.
        section["count"] = _integer(section.get("count"), len(section[item_key]))
        section["count"] = max(section["count"], len(section[item_key]))
        section["count_exact"] = _is_true(section.get("count_exact"))
        if section["count_exact"] and section["count"] != len(section[item_key]):
            section["count_exact"] = False
        a[name] = section
    return a


def derive_filters(a: dict[str, Any]) -> dict[str, Any]:
    """Build counts + evidence flags from an analysis without re-running the VLM.

    Exact absence of a demographic must not be inferred merely because no
    detected person has that label: unknown classifications remain possible.
    """
    people = a.get("people") or {}
    animals = a.get("animals") or {}
    persons = [p for p in people.get("persons", []) if isinstance(p, dict)]
    total = people.get("count")
    exact = people.get("count_exact") is True and type(total) is int and total == len(persons)

    def gender(p: dict) -> str:
        return str(p.get("gender_presentation") or "unknown").lower()

    def age(p: dict) -> str:
        return str(p.get("age_group") or "unknown").lower()

    def count_person(genders: set[str] | None = None, ages: set[str] | None = None) -> int:
        return sum(1 for p in persons if (genders is None or gender(p) in genders)
                   and (ages is None or age(p) in ages))

    child_ages = {"infant", "toddler", "child", "teenager"}
    adult_ages = {"young_adult", "middle_aged", "elderly"}
    age_complete = exact and all(age(p) in child_ages | adult_ages for p in persons)
    child_gender_complete = age_complete and all(gender(p) in {"male", "female"} for p in persons if age(p) in child_ages)
    adult_gender_complete = age_complete and all(gender(p) in {"male", "female"} for p in persons if age(p) in adult_ages)
    mid_gender_complete = age_complete and all(gender(p) in {"male", "female"} for p in persons if age(p) == "middle_aged")
    acount = animals.get("count")
    animal_exact = animals.get("count_exact") is True and type(acount) is int and acount == len(animals.get("items") or [])
    scene = a.get("scene") or {}
    return {
        "people_count": total,
        "people_count_exact": exact,
        "animals_count": acount,
        "animals_count_exact": animal_exact,
        "person_age_complete": age_complete,
        "child_gender_complete": child_gender_complete,
        "adult_gender_complete": adult_gender_complete,
        "middle_aged_gender_complete": mid_gender_complete,
        "male_child_count": count_person({"male"}, child_ages),
        "female_child_count": count_person({"female"}, child_ages),
        "child_count": count_person(ages=child_ages),
        "adult_female_count": count_person({"female"}, adult_ages),
        "middle_aged_male_count": count_person({"male"}, {"middle_aged"}),
        "scene": str(scene.get("location_type") or ""),
        "time_of_day": str(scene.get("time_of_day") or ""),
    }


def prepare_image(source: Path, dst: Path, max_side: int = 1600) -> Path:
    """Convert HEIC/PNG/etc to RGB JPEG, respecting EXIF rotation."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    from PIL import Image, ImageOps
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if im.mode != "RGB":
            if "A" in im.getbands():
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel("A"))
                im = bg
            else:
                im = im.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=92)
    return dst


class TextEmbedder:
    """Embed text using either MLX 4-bit, or the original ModelScope/HF model."""

    def __init__(self, path: Path):
        self.path = validate_model_path(path, "EmbeddingGemma 2")
        cfg = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        self.backend = "mlx" if "quantization" in cfg else "sentence-transformers"
        if self.backend == "mlx":
            import mlx.core as mx
            from transformers import AutoTokenizer
            from mlx_vlm.embedding_loader import load_embedding_model
            self.mx = mx
            self.model = load_embedding_model(self.path)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        else:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "检测到 ModelScope 原始 EmbeddingGemma 2（非 MLX 量化模型）。请先运行：\n"
                    "uv pip install -r requirements-original.txt"
                ) from exc
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = SentenceTransformer(str(self.path), device=device)

    def encode(self, text: str, *, query: bool = False) -> np.ndarray:
        if self.backend == "mlx":
            prefix = "task: search result | query: " if query else "title: none | text: "
            inputs = self.tokenizer(
                [prefix + text], padding=True, truncation=True,
                max_length=1024, return_tensors="np"
            )
            result = self.model(**{k: self.mx.array(v) for k, v in inputs.items()}).text_embeds
            self.mx.eval(result)
            vector = np.asarray(result, dtype=np.float32).reshape(-1)
        else:
            vector = np.asarray(
                self.model.encode(text, prompt_name="SearchQuery" if query else "Document", convert_to_numpy=True),
                dtype=np.float32,
            ).reshape(-1)
        if vector.shape != (VECTOR_DIMENSION,):
            raise RuntimeError(f"向量维度异常：期望 {VECTOR_DIMENSION}，实际 {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("向量包含 NaN/Inf")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise RuntimeError("模型输出全零向量")
        return (vector / norm).astype(np.float32)


class ImageAnalyzer:
    def __init__(self, path: Path):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.path = validate_model_path(path, "Qwen3-VL")
        self.model, self.processor = load(str(self.path))
        self.config = load_config(str(self.path))

    def analyze(self, prepared_image: Path, prompt: str, max_tokens: int = 2200) -> tuple[dict[str, Any], str]:
        self.last_raw = ""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        output = generate(
            self.model, self.processor, formatted, [str(prepared_image)],
            max_tokens=max_tokens, temperature=0.0, verbose=False,
        )
        raw = output.text if hasattr(output, "text") else str(output)
        self.last_raw = raw
        return validate_analysis(read_model_json(raw)), raw
