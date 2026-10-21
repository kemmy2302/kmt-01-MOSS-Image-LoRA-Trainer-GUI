from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def preprocess(path: Path, size: int = 448) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    image = canvas.resize((size, size), Image.Resampling.LANCZOS)
    return np.expand_dims(np.asarray(image, dtype=np.float32)[:, :, ::-1], 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--trigger", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    directory = Path(args.dataset_dir)
    images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit("画像がありません。")

    model_path = hf_hub_download(MODEL_REPO, "model.onnx")
    csv_path = hf_hub_download(MODEL_REPO, "selected_tags.csv")
    labels = pd.read_csv(csv_path)
    names = labels["name"].tolist()
    indexes = list(np.where(labels["category"].isin([0, 4]))[0])
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    written = skipped = failed = 0
    for number, image_path in enumerate(images, 1):
        target = image_path.with_suffix(".txt")
        if target.exists() and not args.overwrite:
            skipped += 1
            print(f"[TAGGER] {number}/{len(images)} skip {image_path.name}", flush=True)
            continue
        try:
            scores = session.run(None, {input_name: preprocess(image_path)})[0][0]
            tags = [names[index].replace("_", " ") for index in indexes if scores[index] >= args.threshold]
            if args.trigger.strip():
                tags.insert(0, args.trigger.strip())
            target.write_text(", ".join(dict.fromkeys(tags)), encoding="utf-8")
            written += 1
            print(f"[TAGGER] {number}/{len(images)} write {image_path.name}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[TAGGER] {number}/{len(images)} error {image_path.name}: {exc}", flush=True)
    print(f"[TAGGER_DONE] written={written} skipped={skipped} failed={failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
