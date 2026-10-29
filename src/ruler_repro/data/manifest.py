from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL 文件不存在：{path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON。") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Any], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"文件已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = asdict(row) if is_dataclass(row) else row
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = asdict(row) if is_dataclass(row) else row
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def latest_by_key(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[str(row[key])] = row
    return result
