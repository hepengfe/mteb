#!/usr/bin/env python3
"""
upload_speechcoco_to_hf_both.py

Build and upload **both training and eval splits in a single run** for SpeechCOCO.

What this script does
- Reads raw SpeechCOCO data (sqlite + wav + images) for *two* splits (train & eval)
- Builds a Hugging Face `DatasetDict` with proper `Audio` and `Image` features
- Pushes both splits to the same Hub repo in a single `push_to_hub` call
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import datasets
from datasets import Dataset, DatasetDict, Features, Value, Audio, Image

try:
    from huggingface_hub import HfApi, HfFolder
except Exception:
    HfApi = None


# --------------------
# Helpers
# --------------------

def pad12(n: int) -> str:
    return f"{int(n):012d}"


def infer_split_from_db(db_path: Path) -> str:
    n = db_path.name.lower()
    if "train_2014" in n:
        return "train2014"
    if "val_2014" in n:
        return "val2014"
    if "train_2017" in n:
        return "train2017"
    if "val_2017" in n:
        return "val2017"
    return "train2014"


def resolve_image_path(image_id: int, images_root: Path, split_hint: str) -> Optional[str]:
    """Try several COCO naming schemes under Images/ folders. Return absolute path or None."""
    candidates = []

    # Prefer year-matched 2014 folder if hint says so
    if split_hint in {"train2014", "val2014"}:
        folder = images_root / split_hint
        prefix = "COCO_" + split_hint
        candidates.append(folder / f"{prefix}_{pad12(image_id)}.jpg")

    # 2017-style fallback (files are just 12-digit id)
    year_folder = "train2017" if "train" in split_hint else "val2017"
    folder2017 = images_root / year_folder
    candidates.append(folder2017 / f"{pad12(image_id)}.jpg")

    # Also try the other 2014 split in case of crossover
    other2014 = "val2014" if split_hint == "train2014" else "train2014"
    folder_other = images_root / other2014
    candidates.append(folder_other / f"COCO_{other2014}_{pad12(image_id)}.jpg")

    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


@dataclass
class Row:
    id: int
    image_id: int
    wav_filename: str
    duration: float
    timecode: str
    disfluency_pos: str
    disfluency_val: str
    speed: float
    text: str
    speaker: str
    gender: str
    nationality: str


# --------------------
# Data generator
# --------------------

def rows_from_sqlite(data_dir: Path) -> Iterator[Row]:
    db_path = None
    for cand in [
        data_dir / "train_2014.sqlite3",
        data_dir / "val_2014.sqlite3",
        data_dir / "train_2017.sqlite3",
        data_dir / "val_2017.sqlite3",
    ]:
        if cand.exists():
            db_path = cand
            break
    if not db_path:
        raise FileNotFoundError("No SpeechCOCO sqlite3 found (expected *_2014.sqlite3 or *_2017.sqlite3)")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    q = (
        "SELECT c.captionID, c.imageID, c.wavFilename, c.duration, c.timecode, c.disfluencyPos, c.disfluencyVal, c.speed, c.text, c.speaker, s.gender, s.nationality "
        "FROM captions c LEFT JOIN speakers s ON c.speaker = s.name"
    )
    try:
        for r in cur.execute(q):
            yield Row(
                id=int(r["captionID"]),
                image_id=int(r["imageID"]),
                wav_filename=str(r["wavFilename"]),
                duration=float(r["duration"]),
                timecode=str(r["timecode"]),
                disfluency_pos=str(r["disfluencyPos"]),
                disfluency_val=str(r["disfluencyVal"]),
                speed=float(r["speed"]),
                text=str(r["text"]),
                speaker=str(r["speaker"]),
                gender=str(r["gender"]) if r["gender"] is not None else "",
                nationality=str(r["nationality"]) if r["nationality"] is not None else "",
            )
    finally:
        conn.close()


# --------------------
# Build HF Dataset (one split)
# --------------------

def build_dataset(data_dir: Path, images_root: Path, *, max_samples: Optional[int] = None) -> Dataset:
    db_path = next(
        (p for p in [
            data_dir / "train_2014.sqlite3",
            data_dir / "val_2014.sqlite3",
            data_dir / "train_2017.sqlite3",
            data_dir / "val_2017.sqlite3",
        ] if p.exists()),
        None,
    )
    if not db_path:
        raise FileNotFoundError("No SpeechCOCO sqlite3 found in data_dir")

    split_name = infer_split_from_db(db_path)
    wav_dir = data_dir / "wav"

    feats = Features({
        "id": Value("int64"),
        "image_id": Value("int64"),
        "audio": Audio(decode=False),
        "image": Image(decode=False),
        "text": Value("string"),
        "duration": Value("float32"),
        "timecode": Value("string"),
        "speaker": Value("string"),
        "gender": Value("string"),
        "nationality": Value("string"),
        "speed": Value("float32"),
        "disfluency_pos": Value("string"),
        "disfluency_val": Value("string"),
    })

    def gen():
        n = 0
        for row in rows_from_sqlite(data_dir):
            audio_path = (wav_dir / row.wav_filename).resolve()
            if not audio_path.exists():
                continue
            image_path = resolve_image_path(row.image_id, images_root, split_name)
            sample = {
                "id": row.id,
                "image_id": row.image_id,
                "audio": {"path": str(audio_path), "bytes": None},
                "image": image_path or None,
                "text": row.text,
                "duration": float(row.duration),
                "timecode": row.timecode,
                "speaker": row.speaker,
                "gender": row.gender,
                "nationality": row.nationality,
                "speed": float(row.speed),
                "disfluency_pos": row.disfluency_pos,
                "disfluency_val": row.disfluency_val,
            }
            yield sample
            n += 1
            if max_samples and n >= max_samples:
                break

    return Dataset.from_generator(gen, features=feats)


# --------------------
# Push
# --------------------

def ensure_login(token: Optional[str]):
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    if HfApi is None:
        return
    if token:
        HfFolder.save_token(token)


def push_to_hub(dset_dict: DatasetDict, repo_id: str, private: bool = False, token: Optional[str] = None):
    ensure_login(token)
    if HfApi is not None:
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
        except Exception:
            pass
    dset_dict.push_to_hub(repo_id, private=private, token=token)


# --------------------
# CLI
# --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data_dir", type=str, required=False, help="Path to TRAIN split dir (contains sqlite + wav/)")
    ap.add_argument("--eval_data_dir", type=str, required=False, help="Path to EVAL/VAL split dir (contains sqlite + wav/)")
    ap.add_argument("--images_root", type=str, default="/media/NAS/NLP/SpeechCoco/Images", help="Root with train2014/ val2014/ train2017/ val2017/")
    ap.add_argument("--repo_id", type=str, default="mteb/SpeechCoco")
    ap.add_argument("--private", type=lambda s: s.lower() in {"1","true","yes"}, default=False)
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    ap.add_argument("--train_max_samples", type=int, default=None)
    ap.add_argument("--eval_max_samples", type=int, default=None)
    ap.add_argument("--push", action="store_true", help="If set, push to Hub")
    ap.add_argument("--dry_run", action="store_true", help="Only build locally and print a summary")
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.train_data_dir and not args.eval_data_dir:
        raise SystemExit("Please provide at least one of --train_data_dir or --eval_data_dir")

    images_root = Path(args.images_root).resolve()

    dsdict: Dict[str, Dataset] = {}

    if args.train_data_dir:
        train_dir = Path(args.train_data_dir).resolve()
        print(f"[INFO] Building TRAIN from {train_dir}")
        train_ds = build_dataset(train_dir, images_root, max_samples=args.train_max_samples)
        dsdict["train"] = train_ds
        print(f"[INFO] TRAIN size: {len(train_ds):,}")
        try:
            print(train_ds[0])
        except Exception:
            pass

    if args.eval_data_dir:
        eval_dir = Path(args.eval_data_dir).resolve()
        print(f"[INFO] Building EVAL from {eval_dir}")
        eval_ds = build_dataset(eval_dir, images_root, max_samples=args.eval_max_samples)
        dsdict["validation"] = eval_ds  # HF convention
        print(f"[INFO] EVAL size: {len(eval_ds):,}")
        try:
            print(eval_ds[0])
        except Exception:
            pass

    if not dsdict:
        raise SystemExit("No datasets were built. Check your paths.")

    dset = DatasetDict(dsdict)
    print(dset)

    if args.dry_run:
        print("[DRY RUN] Not pushing to Hub.")
        return

    if args.push:
        print(f"[INFO] Pushing both splits to hub: {args.repo_id} (private={args.private})")
        push_to_hub(dset, repo_id=args.repo_id, private=args.private, token=args.hf_token)
        print("[DONE] Pushed to Hub.")
    else:
        print("[INFO] Skipping push. Use --push to upload to the Hub.")


if __name__ == "__main__":
    main()
