from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_tree(source: Path, destination: Path) -> int:
    copied = 0
    source_files = [
        path for path in source.rglob("*")
        if path.is_file()
        and path.name != ".gitkeep"
        and destination not in path.parents
    ]
    for source_file in source_files:
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        if destination_file.exists():
            if digest(source_file) != digest(destination_file):
                raise RuntimeError(f"目标文件内容不同，停止迁移：{destination_file}")
            continue
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        if digest(source_file) != digest(destination_file):
            raise RuntimeError(f"复制后校验失败：{destination_file}")
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="非破坏性迁移旧版 Qwen 数据和结果目录")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    model_id = "qwen2_5_7b_instruct"
    mappings = [
        (root / "data" / "generated", root / "data" / "generated" / model_id),
        (root / "data" / "manifests", root / "data" / "manifests" / model_id),
        (root / "results" / "raw", root / "results" / "raw" / model_id),
        (root / "results" / "scored", root / "results" / "scored" / model_id),
    ]
    total = 0
    for source, destination in mappings:
        if source.is_dir():
            total += copy_tree(source, destination)
    print(f"迁移完成：新增 {total} 个文件；旧目录未删除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
