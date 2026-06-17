from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json_preview(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")[:5000000]
    try:
        obj = json.loads(text)
    except Exception:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(next(f)) for _ in range(3)]
    if isinstance(obj, list):
        return obj[:3]
    if isinstance(obj, dict):
        return {k: obj[k] for k in list(obj)[:5]}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/hf")
    parser.add_argument("--out", default="data/metadata/raw_format_preview.json")
    args = parser.parse_args()
    root = Path(args.root)
    previews = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            try:
                size = path.stat().st_size
                if path.suffix == ".jsonl":
                    rows = []
                    with path.open("r", encoding="utf-8") as f:
                        for _, line in zip(range(3), f):
                            rows.append(json.loads(line))
                    previews[str(path)] = {"size": size, "preview": rows}
                else:
                    previews[str(path)] = {"size": size, "preview": read_json_preview(path)}
            except Exception as exc:
                previews[str(path)] = {"size": path.stat().st_size, "error": f"{type(exc).__name__}: {exc}"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(previews, ensure_ascii=False, indent=2)[:200000], encoding="utf-8")
    print(json.dumps({"files": len(previews), "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
