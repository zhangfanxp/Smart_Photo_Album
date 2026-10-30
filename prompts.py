"""Load editable image-analysis prompts from the project-root JSON file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROMPTS_PATH = Path(__file__).with_name("prompts.json")


def _prompt_lines(config: dict[str, Any], key: str) -> str:
    lines = config.get(key)
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        raise ValueError(f"{PROMPTS_PATH} 中的 {key!r} 必须是字符串数组（每行一项）")
    return "\n".join(lines).strip()


def _load_prompts() -> tuple[str, str, str]:
    try:
        config = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到提示词配置文件：{PROMPTS_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"提示词配置不是有效 JSON：{PROMPTS_PATH}（{exc}）") from exc

    if not isinstance(config, dict):
        raise ValueError(f"{PROMPTS_PATH} 顶层必须是 JSON 对象")
    label = config.get("prompt_version")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{PROMPTS_PATH} 缺少非空的 prompt_version")

    analysis_prompt = _prompt_lines(config, "analysis_prompt")
    retry_addendum = _prompt_lines(config, "retry_addendum")
    if not analysis_prompt:
        raise ValueError(f"{PROMPTS_PATH} 中的 analysis_prompt 不能为空")

    # Any semantic prompt edit changes the indexing version automatically.
    # Formatting-only changes to the JSON file do not trigger a full reanalysis.
    canonical = json.dumps(
        {
            "prompt_version": label.strip(),
            "analysis_prompt": analysis_prompt,
            "retry_addendum": retry_addendum,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return analysis_prompt, retry_addendum, f"{label.strip()}-{digest}"


ANALYSIS_PROMPT, _RETRY_ADDENDUM, PROMPT_VERSION = _load_prompts()
RETRY_ANALYSIS_PROMPT = f"{ANALYSIS_PROMPT}\n\n{_RETRY_ADDENDUM}".strip()
