#!/usr/bin/env python3
"""
Quantize trained 3-class YOLO26n (512x512) to ESP-DL `.espdl` format for ESP32-S3
using Espressif's official hybrid INT8/INT16 + TQT + Step-Interpolated INT16 LUT pipeline.

Pipeline:
  1. Export static ONNX with RegMax=1 / ESP_Attention / ESP_Detect patches.
  2. Build hybrid precision dispatch table (Backbone INT8, Neck exits & One2One heads INT16).
  3. Run PTQ Calibration -> TQT scale recovery -> Passive/Alignment -> INT16 LUT Fusion.
  4. Evaluate quantized graph mAP on validation split.
  5. Perform graph surgery (prune aux one2many heads, split Concat into 6 discrete tensors:
     one2one_p3_box, one2one_p3_cls, one2one_p4_box, one2one_p4_cls, one2one_p5_box, one2one_p5_cls).
  6. Export `.espdl` and copy into firmware/main/models/yolo26n_ppe_512_s8_s3.espdl.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from pathlib import Path

# Use non-interactive matplotlib backend so CLI execution never blocks
os.environ.setdefault("MPLBACKEND", "Agg")

ML_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ML_DIR / "vendor"
SCRIPTS_DIR = VENDOR_DIR / "esp_dl_scripts"

sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

import esp_ppq.lib as PFL  # noqa: E402
from esp_ppq.api import get_target_platform  # noqa: E402
from esp_ppq.api.interface import load_onnx_graph  # noqa: E402
from esp_ppq.core import QuantizationVisibility, TargetPlatform  # noqa: E402
from esp_ppq.executor import TorchExecutor  # noqa: E402
from esp_ppq.quantization.optim import (  # noqa: E402
    ParameterQuantizePass,
    PassiveParameterQuantizePass,
    QuantAlignmentPass,
    QuantizeFusionPass,
    QuantizeSimplifyPass,
    RuntimeCalibrationPass,
)

try:  # Optional in esp-ppq >= 1.2.5+
    from esp_ppq.quantization.optim import TrainedQuantizationThresholdPass  # type: ignore
except ImportError:
    TrainedQuantizationThresholdPass = None

# High-sensitivity exit & detection head layers kept in INT16 (per Espressif YOLO26n recipe)
INT16_LAYERS = {
    # Neck Exits
    "/model.16/cv2/conv/Conv", "/model.16/cv2/conv/Conv/Swish",
    "/model.19/cv2/conv/Conv", "/model.19/cv2/conv/Conv/Swish",
    "/model.22/cv2/conv/Conv", "/model.22/cv2/conv/Conv/Swish",
    # Box Heads (one2one_cv2)
    "/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv", "/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv/Swish",
    "/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv", "/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv/Swish",
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
    "/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv", "/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv/Swish",
    "/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv", "/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv/Swish",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv",
    "/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv", "/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv/Swish",
    "/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv", "/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv/Swish",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv",
    # Class Heads (one2one_cv3)
    "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv", "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv/Swish",
    "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv", "/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv/Swish",
    "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv", "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv/Swish",
    "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv", "/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv/Swish",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",
    "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv", "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv/Swish",
    "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv", "/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv/Swish",
    "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv", "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv/Swish",
    "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv", "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv/Swish",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
    "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv", "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv/Swish",
    "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv", "/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv/Swish",
    "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv", "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv/Swish",
    "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv", "/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv/Swish",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv",
}


def build_qat_config(args: argparse.Namespace):
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )
    out_dir = (ML_DIR / "output" / f"ppe_{args.imgsz}_s8_{args.platform}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class QATConfig:
        IMG_SZ = args.imgsz
        DEVICE = device
        DATA_YAML_FILE = str(args.data.resolve())
        BATCH_SIZE = args.batch_size
        CALIB_MAX_IMAGES = args.calib_max_images
        CALIB_VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        DATA_FALLBACK_PATH = str((args.data.resolve().parent / "images" / "train"))

        CALIB_STEPS = args.calib_steps
        QUANT_CALIB_METHOD = args.calib_method
        QUANT_ALIGNMENT = "Align to Output"
        TARGET_PLATFORM = get_target_platform("esp32" + args.platform, 8)
        INT16_LUT_STEP = args.lut_step

        TQT_STEPS = args.tqt_steps
        TQT_LR = 1e-5
        TQT_INT_LAMBDA = 0.25
        TQT_BLOCK_SIZE = 4
        TQT_COLLECTING_DEVICE = "cpu"
        VAL_PLOT_MAX_BATCHES = 0

        BASE_DIR = str(ML_DIR)
        MODEL_NAME = "yolo26n_ppe"
        PT_FILE = str(args.weights.resolve())
        ONNX_FILE = f"{MODEL_NAME}_{args.imgsz}_export.onnx"
        ESPDL_OUTPUT_DIR = str(out_dir)
        ONNX_PATH = str(out_dir / ONNX_FILE)

    return QATConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize YOLO26n PPE model to .espdl for ESP32-S3.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ML_DIR / "weights" / "yolo26n_ppe.pt",
        help="Path to trained best .pt weights.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ML_DIR / "datasets" / "ppe_3class" / "data.yaml",
        help="Path to 3-class data.yaml.",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Optional custom calibration images dir (e.g., CoreS3 GC0308 captures).",
    )
    parser.add_argument("--imgsz", type=int, default=512, help="Model input resolution (square).")
    parser.add_argument("--platform", default="s3", choices=["s3", "p4", "s31"], help="ESP target chip.")
    parser.add_argument("--batch-size", type=int, default=8, help="Calibration/validation batch size.")
    parser.add_argument("--calib-steps", type=int, default=32, help="PTQ calibration batches.")
    parser.add_argument("--calib-max-images", type=int, default=512, help="Max calibration images.")
    parser.add_argument("--calib-method", default="percentile", choices=["percentile", "kl", "mse"])
    parser.add_argument("--tqt-steps", type=int, default=100, help="TQT block optimization steps (0 to skip).")
    parser.add_argument("--lut-step", type=int, default=32, help="INT16 Swish LUT step size.")
    parser.add_argument("--device", default="auto", help="Quantization device: auto (cuda/cpu), cpu, cuda.")
    parser.add_argument("--skip-val", action="store_true", help="Skip quantized mAP evaluation.")
    parser.add_argument(
        "--firmware-model-dst",
        type=Path,
        default=ML_DIR.parent / "firmware" / "main" / "models" / "yolo26n_ppe_512_s8_s3.espdl",
        help="Destination path to copy exported .espdl model for firmware build.",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    # Inject QATConfig before importing vendored ESP-DL scripts
    QATConfig = build_qat_config(args)
    cfg_mod = types.ModuleType("config")
    cfg_mod.QATConfig = QATConfig
    sys.modules["config"] = cfg_mod

    import esp_ppq_lut as esp_lut  # noqa: E402
    from dataset import SubsetCaliDataset, get_calibration_loader  # noqa: E402
    from esp_ppq_lut.passes import EspdlLUTFusionPass  # noqa: E402
    from esp_ppq_patch import apply_esp_ppq_patches  # noqa: E402
    from esp_ppq_patch_2 import apply_addlut_patch  # noqa: E402
    from notebook_helpers import (  # noqa: E402
        eval_espdl_model,
        extract_model_meta,
        prepare_onnx,
        prune_graph_safely,
    )
    from trainer import QATTrainer  # noqa: E402
    from utils import get_exclusive_ancestors, register_mod_op, seed_everything  # noqa: E402

    print(
        f"[quantize] Platform=esp32{args.platform} | imgsz={args.imgsz} | "
        f"device={QATConfig.DEVICE} | tqt_steps={args.tqt_steps} | lut_step={args.lut_step}"
    )

    esp_lut.initialize(step=QATConfig.INT16_LUT_STEP, verbose=True)
    seed_everything(1234)
    register_mod_op()
    apply_esp_ppq_patches()
    apply_addlut_patch()

    # 1. Export static ONNX with RegMax=1 / ESP_Attention / ESP_Detect patches
    prepare_onnx()
    model_meta = extract_model_meta()

    # 2. Load ONNX into ESP-PPQ and split aux (one2many) vs main (one2one) heads
    print("[quantize] Loading ONNX graph into ESP-PPQ...")
    graph = load_onnx_graph(onnx_import_file=QATConfig.ONNX_PATH)

    output_names = list(graph.outputs.keys())
    aux_ops, main_ops = set(), set()
    if len(output_names) >= 6:
        aux_outputs = output_names[0:3]
        main_outputs = output_names[3:6]
        aux_ops = get_exclusive_ancestors(graph, aux_outputs, main_outputs)
        main_ops = get_exclusive_ancestors(graph, main_outputs, aux_outputs)

    quantizer = PFL.Quantizer(platform=QATConfig.TARGET_PLATFORM, graph=graph)
    dispatching_table = PFL.Dispatcher(graph=graph, method="conservative").dispatch(
        quantizer.quant_operation_types
    )
    for opname, plat in dispatching_table.items():
        if plat == TargetPlatform.UNSPECIFIED:
            dispatching_table[opname] = TargetPlatform(quantizer.target_platform)
    for op in aux_ops:
        if op.name in dispatching_table:
            dispatching_table[op.name] = TargetPlatform.FP32

    # 3. Assign INT16 hybrid precision to sensitive neck exits + one2one box/cls heads
    int16_platform = get_target_platform("esp32" + args.platform, 16)
    int16_matched = 0
    for op in graph.operations.values():
        if op.name in dispatching_table and op.name in INT16_LAYERS:
            dispatching_table[op.name] = int16_platform
            int16_matched += 1
    print(f"[quantize] Dispatched {int16_matched}/{len(INT16_LAYERS)} sensitive ops to INT16.")

    # Keep output Concat ops FP32 prior to graph surgery
    fp32_concats = {"/model.23/Concat_5", "/model.23/Concat_3", "/model.23/Concat_4"}
    for op in main_ops:
        if op.name in fp32_concats or op.type == "Concat":
            if any(out.name in graph.outputs for out in op.outputs):
                dispatching_table[op.name] = TargetPlatform.FP32

    for op in graph.operations.values():
        quantizer.quantize_operation(op_name=op.name, platform=dispatching_table[op.name])

    # 4. Prepare calibration dataloader
    data_cfg = check_det_dataset(QATConfig.DATA_YAML_FILE)
    if args.calib_dir and args.calib_dir.is_dir():
        print(f"[quantize] Using custom calibration folder: {args.calib_dir}")
        cali_dataset = SubsetCaliDataset(str(args.calib_dir), max_images=QATConfig.CALIB_MAX_IMAGES)
        cali_loader = DataLoader(cali_dataset, batch_size=QATConfig.BATCH_SIZE, shuffle=False)
    else:
        cali_loader = get_calibration_loader(data_cfg)

    actual_calib_steps = min(QATConfig.CALIB_STEPS, max(1, len(cali_loader)))
    print(f"[quantize] Calibration images: {len(cali_loader.dataset)} | steps={actual_calib_steps}")

    # 5. Run PTQ + optional TQT + LUT fusion pipeline
    executor = TorchExecutor(graph=graph, device=QATConfig.DEVICE)
    dummy_input = torch.zeros([1, 3, QATConfig.IMG_SZ, QATConfig.IMG_SZ], device=QATConfig.DEVICE)
    executor.tracing_operation_meta(inputs=dummy_input)

    passes = [
        QuantizeSimplifyPass(),
        QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
        ParameterQuantizePass(),
        RuntimeCalibrationPass(method=QATConfig.QUANT_CALIB_METHOD),
    ]
    if QATConfig.TQT_STEPS > 0 and TrainedQuantizationThresholdPass is not None:
        passes.append(
            TrainedQuantizationThresholdPass(
                steps=QATConfig.TQT_STEPS,
                lr=QATConfig.TQT_LR,
                int_lambda=QATConfig.TQT_INT_LAMBDA,
                block_size=QATConfig.TQT_BLOCK_SIZE,
                collecting_device=QATConfig.TQT_COLLECTING_DEVICE,
            )
        )
    elif QATConfig.TQT_STEPS > 0:
        print(
            "[quantize] Note: TrainedQuantizationThresholdPass not in installed esp-ppq; "
            "proceeding with hybrid INT8/INT16 PTQ + INT16 LUT."
        )
    passes.extend(
        [
            PassiveParameterQuantizePass(clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE),
            QuantAlignmentPass(elementwise_alignment=QATConfig.QUANT_ALIGNMENT),
            EspdlLUTFusionPass(target_ops=["Swish"], lut_step=QATConfig.INT16_LUT_STEP),
        ]
    )

    pipeline = PFL.Pipeline(passes)
    pipeline.optimize(
        calib_steps=actual_calib_steps,
        collate_fn=(lambda x: x.type(torch.float).to(QATConfig.DEVICE)),
        graph=graph,
        dataloader=cali_loader,
        executor=executor,
    )
    print("[quantize] Optimization pipeline complete.")

    # 6. Validate quantized graph before tearing Concat nodes
    if not args.skip_val:
        print("[quantize] Evaluating bit-exact quantized graph on validation set...")
        eval_wrapper = QATTrainer(graph=graph, model_meta=model_meta, device=QATConfig.DEVICE)
        val_map = eval_wrapper.eval()
        print(f"[quantize] Quantized Val mAP50-95: {val_map:.4f}")

    # 7. Graph Surgery: prune aux heads & split one2one_p{3,4,5} Concat into box/cls pairs
    print("[quantize] Performing graph surgery for 6 discrete ESP-DL outputs...")
    output_names = list(graph.outputs.keys())
    if len(output_names) >= 6:
        for name in output_names[0:3]:
            graph.outputs.pop(name, None)
        prune_graph_safely(graph)

    targets = ["one2one_p3", "one2one_p4", "one2one_p5"]
    collected_outputs = {}
    for target_name in targets:
        if target_name not in graph.outputs:
            raise RuntimeError(f"Expected graph output '{target_name}' missing before surgery.")
        original_output_var = graph.variables[target_name]
        producer = original_output_var.source_op
        if not producer or producer.type != "Concat":
            raise RuntimeError(f"Expected Concat producer for '{target_name}', got {producer}")

        box_var, cls_var = None, None
        for input_var in producer.inputs:
            dims = input_var.shape
            if dims is not None and len(dims) >= 2:
                # Check channel dim explicitly (index 1 in NCHW) so nc=3/4 never collides with H/W
                ch_dim = int(dims[1])
                if ch_dim == 4 and box_var is None:
                    box_var = input_var
                elif ch_dim == int(model_meta["nc"]) and cls_var is None:
                    cls_var = input_var

        if not box_var or not cls_var:
            raise RuntimeError(
                f"Failed to identify box(C=4) and cls(C={model_meta['nc']}) inputs of {producer.name}: "
                f"{[(v.name, v.shape) for v in producer.inputs]}"
            )

        for var, new_name in [(box_var, f"{target_name}_box"), (cls_var, f"{target_name}_cls")]:
            old_name = var.name
            if old_name in graph.variables:
                graph.variables.pop(old_name)
            var._name = new_name
            graph.variables[new_name] = var
            collected_outputs[new_name] = var

        graph.outputs.pop(target_name)
        graph.remove_operation(producer, keep_coherence=False)
        for var in producer.inputs:
            if producer in var.dest_ops:
                var.dest_ops.remove(producer)

    final_output_list = [
        "one2one_p3_box", "one2one_p3_cls",
        "one2one_p4_box", "one2one_p4_cls",
        "one2one_p5_box", "one2one_p5_cls",
    ]
    graph.outputs.clear()
    for name in final_output_list:
        graph.outputs[name] = collected_outputs[name]
    prune_graph_safely(graph)
    print(f"[quantize] Final graph outputs: {list(graph.outputs.keys())}")

    # 8. Save one bit-exact preview image from validation split
    val_dir = Path(data_cfg["val"])
    sample_imgs = sorted(
        p for p in val_dir.glob("*") if p.suffix.lower() in QATConfig.CALIB_VALID_EXTENSIONS
    )
    if sample_imgs:
        preview_dir = ML_DIR / "results"
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Monkey-patch plt.show to no-op for headless CLI
        plt.show = lambda *a, **kw: None
        preds, saved_path = eval_espdl_model(
            test_image_path=str(sample_imgs[0]),
            graph=graph,
            target_img_sz=QATConfig.IMG_SZ,
            data_yaml=QATConfig.DATA_YAML_FILE,
            platform=args.platform,
            conf_thresh=0.25,
            output_dir=str(preview_dir),
        )
        print(f"[quantize] Preview saved to {saved_path} ({len(preds)} detections)")

    # 9. Export .espdl and copy to firmware
    final_espdl_path = Path(QATConfig.ESPDL_OUTPUT_DIR) / f"{QATConfig.MODEL_NAME}_{args.imgsz}_s8_{args.platform}.espdl"
    exporter = PFL.Exporter(platform=QATConfig.TARGET_PLATFORM)
    exporter.export(str(final_espdl_path), graph=graph, int16_lut_step=QATConfig.INT16_LUT_STEP)
    size_mb = final_espdl_path.stat().st_size / (1024 * 1024)
    print(f"[quantize] Exported ESP-DL model -> {final_espdl_path} ({size_mb:.2f} MB)")

    if args.firmware_model_dst:
        args.firmware_model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_espdl_path, args.firmware_model_dst)
        print(f"[quantize] Copied model to firmware -> {args.firmware_model_dst}")


if __name__ == "__main__":
    main()
