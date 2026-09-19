#!/usr/bin/env python3
"""
fetch_datasets - build the task files that ship with benchmark.py.

This is the only part of the harness that needs the HuggingFace `datasets`
library, and students never have to run it: benchmark.py reads the resulting
files straight off disk.

    pip install datasets
    python fetch_datasets.py --split public
    python fetch_datasets.py --split private --n-hellaswag 200 --n-arc 200

Writes:
    data/<split>/hellaswag.jsonl
    data/<split>/arc_easy.jsonl
    data/<split>/ifeval.json

Caching more rows than the suite uses is deliberate: benchmark.py picks its
items with a fixed seed, so a larger pool can be re-sampled for a held-out
split without touching the code.

NOTE ON THE FILENAME: this file is *not* called datasets.py on purpose. The
script's own directory is first on sys.path, so a local datasets.py would
shadow the installed `datasets` package and `from datasets import
load_dataset` below would import this file instead of the library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

HF_SPECS = {
    # name        (repo,                config,     split)
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
}

DEFAULT_IFEVAL = [
    {"id": "if01",
     "prompt": "Describe the ocean in exactly 3 sentences. Do not use any commas.",
     "checks": [{"type": "sentence_count", "args": [3], "label": "3 sentences"},
                {"type": "forbid_chars", "args": [","], "label": "no commas"}]},
    {"id": "if02",
     "prompt": "Write a short paragraph about cats. Your entire response must be in "
               "lowercase letters. No capital letters are allowed.",
     "checks": [{"type": "all_lowercase", "label": "all lowercase"},
                {"type": "word_count_between", "args": [15, 120], "label": "15-120 words"}]},
    {"id": "if03",
     "prompt": "List exactly 4 kitchen utensils. Use a bulleted list where every line "
               "starts with the '*' character. Output nothing else.",
     "checks": [{"type": "bullet_count", "args": [4], "label": "exactly 4 bullets"}]},
    {"id": "if04",
     "prompt": "Return a JSON object describing a book. It must contain the keys "
               "\"title\", \"author\" and \"year\". Output only the JSON.",
     "checks": [{"type": "valid_json_with_keys",
                 "args": [["title", "author", "year"]], "label": "valid json + keys"}]},
    {"id": "if05",
     "prompt": "Explain why the sky is blue in between 40 and 80 words. Finish your "
               "reply with the exact phrase: That is the answer.",
     "checks": [{"type": "word_count_between", "args": [40, 80], "label": "40-80 words"},
                {"type": "ends_with", "args": ["That is the answer."],
                 "label": "ends with phrase"}]},
    {"id": "if06",
     "prompt": "Write a product blurb for a bicycle. Wrap your entire response in "
               "double quotation marks.",
     "checks": [{"type": "wrapped_in_quotes", "label": "wrapped in quotes"},
                {"type": "word_count_between", "args": [10, 100], "label": "10-100 words"}]},
    {"id": "if07",
     "prompt": "Give me a title for an article about coffee. The title must be wrapped "
               "in double angular brackets, like <<this>>.",
     "checks": [{"type": "title_in_angle_brackets", "label": "<<title>>"}]},
    {"id": "if08",
     "prompt": "Write about the seasons in exactly 2 paragraphs, separated by a blank "
               "line. The word \"weather\" must appear at least 3 times.",
     "checks": [{"type": "paragraph_count", "args": [2], "label": "2 paragraphs"},
                {"type": "contains_at_least", "args": ["weather", 3],
                 "label": "'weather' x3"}]},
    {"id": "if09",
     "prompt": "Answer in ALL CAPITAL LETTERS: what is the capital city of Japan?",
     "checks": [{"type": "all_uppercase", "label": "all caps"},
                {"type": "contains_at_least", "args": ["TOKYO", 1], "label": "correct"}]},
    {"id": "if10",
     "prompt": "Begin your response with the word \"Certainly\". Then describe a rainy "
               "day. Do not use the words \"wet\" or \"umbrella\".",
     "checks": [{"type": "starts_with", "args": ["Certainly"],
                 "label": "starts with Certainly"},
                {"type": "forbid_words", "args": [["wet", "umbrella"]],
                 "label": "forbidden words absent"}]},
]


def fetch_ranking(name: str, limit: int, out_dir: Path, offset: int = 0) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("this script needs the HuggingFace datasets library:\n"
                         "    pip install datasets")

    repo, config, hf_split = HF_SPECS[name]
    ds = load_dataset(repo, config, split=f"{hf_split}[{offset}:{offset + limit}]")
    rows = [dict(r) for r in ds]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows):>4} rows -> {path}")
    return len(rows)


def write_ifeval(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ifeval.json"
    path.write_text(json.dumps(DEFAULT_IFEVAL, indent=2), encoding="utf-8")
    print(f"  wrote {len(DEFAULT_IFEVAL):>4} items -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="public", help="data/<split>/ subdirectory")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--n-hellaswag", type=int, default=200)
    ap.add_argument("--n-arc", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many source rows; use a non-zero offset to "
                         "build a private split that cannot overlap the public one")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    out_dir = Path(args.data_dir).expanduser() / args.split
    existing = [p for p in (out_dir / "hellaswag.jsonl", out_dir / "arc_easy.jsonl",
                            out_dir / "ifeval.json") if p.exists()]
    if existing and not args.force:
        print(f"refusing to overwrite existing files in {out_dir}:", file=sys.stderr)
        for p in existing:
            print(f"    {p.name}", file=sys.stderr)
        print("re-run with --force to replace them.", file=sys.stderr)
        return 1

    print(f"building split '{args.split}' in {out_dir}")
    fetch_ranking("hellaswag", args.n_hellaswag, out_dir, args.offset)
    fetch_ranking("arc_easy", args.n_arc, out_dir, args.offset)
    write_ifeval(out_dir)
    print("done. benchmark.py can now run without the datasets library.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
