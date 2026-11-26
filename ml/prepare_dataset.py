#!/usr/bin/env python3
"""
Download Ultralytics construction-ppe dataset and remap its 12 classes into a
clean 3-class helmet-monitoring dataset:
    0: helmet     (original 0: helmet)
    1: no_helmet  (original 6: no_helmet)
    2: person     (original 5: person)

All other classes (gloves, vest, boots, goggles, forklift, etc.) are filtered out.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from ultralytics.data.utils import check_det_dataset

# Target 3-class mapping (resolved dynamically by name from construction-ppe.yaml!)
TARGET_NAME_TO_NEW_ID = {
    "helmet": 0,
    "no_helmet": 1,
    "person": 2,
}

NEW_CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet",
    2: "person",
}


def build_class_remap(src_names: dict[int, str] | list[str]) -> dict[int, int]:
    if isinstance(src_names, list):
        items = enumerate(src_names)
    else:
        items = src_names.items()
    remap: dict[int, int] = {}
    for orig_id, raw_name in items:
        key = str(raw_name).strip().lower()
        if key in TARGET_NAME_TO_NEW_ID:
            remap[int(orig_id)] = TARGET_NAME_TO_NEW_ID[key]
    missing = set(TARGET_NAME_TO_NEW_ID.values()) - set(remap.values())
    if missing:
        raise RuntimeError(
            f"Could not find required target classes in source YAML names={src_names}; resolved remap={remap}"
        )
    print(f"[prepare_dataset] Dynamically resolved class remap: {remap} (from {src_names})")
    return remap


def resolve_split_images(dataset_root: Path, split_entry: str | list[str] | None) -> list[Path]:
    if not split_entry:
        return []
    entries = [split_entry] if isinstance(split_entry, str) else list(split_entry)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    for entry in entries:
        p = Path(entry)
        if not p.is_absolute():
            p = dataset_root / p
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in image_exts:
                    images.append(child)
        elif p.is_file() and p.suffix == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                img_p = Path(line)
                if not img_p.is_absolute():
                    img_p = dataset_root / img_p
                if img_p.exists():
                    images.append(img_p)
    return images


def image_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    # Replace the last occurrence of 'images' with 'labels'
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def remap_label_file(
    src_label: Path, dst_label: Path, class_remap: dict[int, int], counts: Counter[int]
) -> int:
    kept_lines: list[str] = []
    if src_label.exists():
        for raw_line in src_label.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            try:
                old_cls = int(float(tokens[0]))
            except ValueError:
                continue
            if old_cls in class_remap:
                new_cls = class_remap[old_cls]
                counts[new_cls] += 1
                kept_lines.append(" ".join([str(new_cls)] + tokens[1:]))
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return len(kept_lines)


def load_builtin_ppe_names() -> dict[int, str]:
    import ultralytics

    cfg_path = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "construction-ppe.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return cfg["names"]


def prepare_dataset(output_dir: Path, force: bool = False, source_dir: Path | None = None) -> Path:
    output_dir = output_dir.resolve()
    yaml_out = output_dir / "data.yaml"
    if yaml_out.exists() and not force:
        print(f"[prepare_dataset] Dataset already prepared at {yaml_out} (use --force to rebuild).")
        return yaml_out

    local_default_src = Path(__file__).resolve().parent / "datasets" / "construction-ppe"
    resolved_src = source_dir.resolve() if source_dir else (local_default_src if local_default_src.is_dir() else None)

    if resolved_src and (resolved_src / "images").is_dir():
        print(f"[prepare_dataset] Using pre-downloaded local source dataset: {resolved_src}")
        src_root = resolved_src
        local_yaml = src_root / "data.yaml"
        if local_yaml.is_file():
            local_cfg = yaml.safe_load(local_yaml.read_text(encoding="utf-8"))
            names = local_cfg["names"]
            print(f"[prepare_dataset] Loaded authoritative class names from {local_yaml}")
        else:
            names = load_builtin_ppe_names()
        src_info: dict[str, object] = {"path": str(src_root), "names": names}
        for sp in ("train", "val", "test"):
            if (src_root / "images" / sp).is_dir():
                src_info[sp] = f"images/{sp}"
    else:
        print("[prepare_dataset] Ensuring Ultralytics construction-ppe dataset is downloaded...")
        src_info = check_det_dataset("construction-ppe.yaml")
        src_root = Path(src_info["path"]).resolve()
        local_yaml = src_root / "data.yaml"
        if local_yaml.is_file():
            local_cfg = yaml.safe_load(local_yaml.read_text(encoding="utf-8"))
            if "names" in local_cfg:
                src_info["names"] = local_cfg["names"]
                print(f"[prepare_dataset] Overrode names with authoritative {local_yaml}")
    class_remap = build_class_remap(src_info["names"])
    print(f"[prepare_dataset] Source dataset root: {src_root}")

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    splits = ["train", "val", "test"]
    available_splits: dict[str, str] = {}

    for split in splits:
        if split not in src_info or not src_info[split]:
            continue
        src_images = resolve_split_images(src_root, src_info[split])
        if not src_images:
            continue

        img_out_dir = output_dir / "images" / split
        lbl_out_dir = output_dir / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        cls_counts: Counter[int] = Counter()
        empty_count = 0

        for idx, src_img in enumerate(src_images):
            dst_img = img_out_dir / src_img.name
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            # Symlink to avoid duplicating disk space (fallback to copy if needed)
            try:
                os.symlink(src_img.resolve(), dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

            src_lbl = image_to_label_path(src_img)
            dst_lbl = lbl_out_dir / f"{src_img.stem}.txt"
            kept = remap_label_file(src_lbl, dst_lbl, class_remap, cls_counts)
            if kept == 0:
                empty_count += 1

        available_splits[split] = f"images/{split}"
        summary = ", ".join(
            f"{NEW_CLASS_NAMES[cid]}={cls_counts[cid]}" for cid in sorted(NEW_CLASS_NAMES)
        )
        print(
            f"[prepare_dataset] Split '{split}': {len(src_images)} images "
            f"(bg/empty={empty_count}) | boxes: {summary}"
        )

    data_cfg = {
        "path": str(output_dir),
        "train": available_splits.get("train", "images/train"),
        "val": available_splits.get("val", "images/val"),
        "nc": len(NEW_CLASS_NAMES),
        "names": NEW_CLASS_NAMES,
    }
    if "test" in available_splits:
        data_cfg["test"] = available_splits["test"]

    yaml_out.write_text(yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[prepare_dataset] Wrote 3-class dataset config: {yaml_out}")
    return yaml_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 3-class helmet PPE dataset from construction-ppe.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "ppe_3class",
        help="Destination directory for remapped 3-class dataset.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional path to pre-downloaded construction-ppe directory (auto-detects ml/datasets/construction-ppe).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild dataset even if it already exists.")
    args = parser.parse_args()
    prepare_dataset(args.output, force=args.force, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
