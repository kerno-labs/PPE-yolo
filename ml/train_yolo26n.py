#!/usr/bin/env python3
"""
Train YOLO26n (512x512) on the 3-class PPE helmet dataset (helmet, no_helmet, person).
Automatically selects Apple Silicon MPS, NVIDIA CUDA, or CPU.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import unwrap_model


def _to_cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_tree(v) for v in obj)
    return obj


def _first_device(obj) -> torch.device | None:
    if isinstance(obj, torch.Tensor):
        return obj.device
    if isinstance(obj, dict):
        for v in obj.values():
            d = _first_device(v)
            if d is not None:
                return d
    if isinstance(obj, (list, tuple)):
        for v in obj:
            d = _first_device(v)
            if d is not None:
                return d
    return None


def patch_mps_e2e_loss() -> None:
    """Compute dynamic-shape E2ELoss (TAL + fg_mask CIoU + BCE) on CPU when training on Apple MPS.

    Moving only the compact detection head outputs (~2 MB) to CPU is differentiable
    in PyTorch autograd, keeps all 260 backbone/neck/head layers on the MPS GPU with
    100% static tensor shapes, eliminates per-batch MPSGraph shader recompilations,
    prevents Metal graph-cache memory growth, and yields a ~4-5x wall-clock speedup.
    """
    orig_call = E2ELoss.__call__

    def mps_safe_call(self, preds, batch):
        dev = _first_device(preds)
        if dev is not None and dev.type == "mps":
            for sub in (self.one2many, self.one2one):
                if sub.device.type != "cpu":
                    sub.device = torch.device("cpu")
                    sub.proj = sub.proj.cpu()
                    sub.stride = sub.stride.cpu()
                    sub.bbox_loss = sub.bbox_loss.cpu()
            loss, loss_items = orig_call(self, _to_cpu_tree(preds), _to_cpu_tree(batch))
            return loss.to(dev), loss_items.to(dev)
        return orig_call(self, preds, batch)

    E2ELoss.__call__ = mps_safe_call


class MPSFixedShapeTrainer(DetectionTrainer):
    """Avoids Apple MPSGraph shader recompilation caused by variable `rect=True` val shapes."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 8, rank: int = 0, mode: str = "train"):
        if mode == "val":
            batch_size = min(batch_size, 8)
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26n 512x512 on 3-class PPE dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "ppe_3class" / "data.yaml",
        help="Path to 3-class data.yaml.",
    )
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained YOLO26 checkpoint.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size (square).")
    parser.add_argument("--batch", type=int, default=10, help="Batch size (8-12 is optimal on Apple MPS).")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cpu, or CUDA index.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer: auto (AdamW on MPS, MuSGD on CUDA), AdamW, SGD, MuSGD.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 recommended with cache=ram on macOS).")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt in runs/yolo26n_ppe_<imgsz>.")
    parser.add_argument(
        "--out-weights",
        type=Path,
        default=Path(__file__).resolve().parent / "weights" / "yolo26n_ppe.pt",
        help="Path to copy best trained weights.",
    )
    args = parser.parse_args()

    os.environ["COMET_MODE"] = "disabled"
    os.environ["WANDB_MODE"] = "disabled"

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}. Run prepare_dataset.py first."
        )

    device = pick_device(args.device)
    if args.optimizer == "auto":
        # MuSGD's bf16 Newton-Schulz step hits an MPSGraph f32*bf16 assertion on PyTorch 2.6 MPS
        optimizer = "AdamW" if device == "mps" else "MuSGD"
    else:
        optimizer = args.optimizer
    lr0 = 0.0015 if optimizer == "AdamW" else 0.005
    use_amp = device not in ("mps", "cpu")

    ml_dir = Path(__file__).resolve().parent
    project_dir = ml_dir / "runs"
    run_name = f"yolo26n_ppe_{args.imgsz}"

    print(
        f"[train] Starting YOLO26n training | data={args.data} | imgsz={args.imgsz} | "
        f"epochs={args.epochs} | batch={args.batch} | device={device} | optimizer={optimizer} | lr0={lr0}"
    )

    last_pt = project_dir / run_name / "weights" / "last.pt"
    if args.resume and last_pt.exists():
        print(f"[train] Resuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO(args.model)

    if device == "mps":
        patch_mps_e2e_loss()

        def _mps_cleanup(_t):
            gc.collect()
            torch.mps.empty_cache()

        model.add_callback("on_train_epoch_end", _mps_cleanup)
        model.add_callback("on_fit_epoch_end", _mps_cleanup)

    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        optimizer=optimizer,
        lr0=lr0,
        lrf=0.01,
        amp=use_amp,
        cache="ram",
        seed=42,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        resume=bool(args.resume and last_pt.exists()),
    )
    if device == "mps":
        train_kwargs["trainer"] = MPSFixedShapeTrainer
    model.train(**train_kwargs)

    best_pt = project_dir / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Expected trained weights not found at {best_pt}")

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, args.out_weights)
    print(f"[train] Saved best weights -> {args.out_weights}")

    # Final validation summary on best.pt (square fixed-shape matching CoreS3 512x512 inference)
    best_model = YOLO(str(args.out_weights))
    val_bs = min(args.batch, 8) if device == "mps" else args.batch
    metrics = best_model.val(
        data=str(args.data), imgsz=args.imgsz, batch=val_bs, rect=False, device=device, plots=False
    )
    print(
        f"[train] FP32 Best Val | mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f}"
    )


if __name__ == "__main__":
    main()
