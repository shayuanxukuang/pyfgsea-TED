"""Build a deterministic SHA256 manifest for the TED release tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_MANIFEST.tsv"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".release_validation",
    ".venv",
    "__pycache__",
    "target",
    "dist",
    "build",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(ROOT.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        rows.append((relative.as_posix(), path.stat().st_size, sha256(path)))
    content = ["path\tbytes\tsha256"]
    content.extend(f"{path}\t{size}\t{digest}" for path, size, digest in rows)
    OUTPUT.write_text("\n".join(content) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(rows)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
