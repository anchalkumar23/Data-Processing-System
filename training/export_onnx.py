"""
training/export_onnx.py
────────────────────────
Exports the trained PyTorch model to ONNX format for production serving.

Why ONNX?
  • Framework-agnostic: the exported model can be run with ONNX Runtime,
    TensorRT, OpenVINO, or CoreML — without a PyTorch dependency in prod.
  • Typically 2-4× faster inference than PyTorch eager mode on CPU.
  • Supports hardware-specific optimisations (GPU, NPU, edge devices).
  • opset 17 is the modern standard as of 2024.

After export, we validate the ONNX graph and run a quick numerical
sanity check to confirm the outputs match the PyTorch model.

Run:
    python -m training.export_onnx
    python -m training.export_onnx --checkpoint checkpoints/best_model.pth
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

# onnx (for graph validation) is optional — it requires C++ compilation on
# Python 3.13 Windows and has no pre-built wheel there.  Inference via
# onnxruntime works fine without it; we just skip the structural check.
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    log.warning("'onnx' package not installed — graph validation will be skipped. "
                "Install it manually if you want structural checks: pip install onnx")

from training.model import build_model, _get_device, load_checkpoint

log = logging.getLogger(__name__)


def export_to_onnx(
    model:        torch.nn.Module,
    output_path:  Path,
    input_shape:  tuple = (1, 3, 224, 224),
    opset:        int   = 17,
    dynamic_axes: bool  = True,
) -> None:
    """
    Exports a PyTorch model to an ONNX file.

    Args:
        model:        Trained model in eval mode.
        output_path:  Where to save the .onnx file.
        input_shape:  Example input shape (B, C, H, W).
        opset:        ONNX opset version.
        dynamic_axes: If True, batch size can vary at inference time.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Create a dummy input on the same device as the model
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    # Dynamic axes let us pass any batch size at runtime — critical for
    # the real-time inference pipeline where batch size varies per request
    axes = None
    if dynamic_axes:
        axes = {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    log.info(f"Exporting model to ONNX (opset {opset}) → {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version      = opset,
        input_names        = ["input"],
        output_names       = ["output"],
        dynamic_axes       = axes,
        do_constant_folding = True,  # fuse constant subgraphs for speed
        export_params      = True,
    )

    log.info(f"Export complete: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def validate_onnx(onnx_path: Path) -> None:
    """
    Checks the ONNX graph for structural errors using the official checker.
    Skipped gracefully if the 'onnx' package is not installed (no Python 3.13
    Windows wheel available — install manually if needed).
    """
    if not ONNX_AVAILABLE:
        log.warning("Skipping ONNX graph validation ('onnx' package not installed).")
        return

    log.info("Validating ONNX graph structure...")
    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)
    log.info("✓ ONNX validation passed")


def numerical_sanity_check(
    torch_model: torch.nn.Module,
    onnx_path:   Path,
    tolerance:   float = 1e-4,
) -> None:
    """
    Runs the same dummy input through both the PyTorch model and the
    ONNX Runtime session, then checks that outputs match within `tolerance`.

    This catches numerical differences introduced by the export process
    (e.g. fused ops changing computation order slightly).
    """
    log.info("Running numerical sanity check (PyTorch vs ONNX Runtime)...")

    dummy = np.random.randn(2, 3, 224, 224).astype(np.float32)

    # PyTorch output
    torch_model.eval()
    with torch.no_grad():
        pt_out = torch_model(torch.from_numpy(dummy)).numpy()

    # ONNX Runtime output
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_out = sess.run(["output"], {"input": dummy})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    log.info(f"  Max output difference: {max_diff:.2e}  (tolerance: {tolerance:.2e})")

    if max_diff > tolerance:
        raise RuntimeError(
            f"ONNX sanity check failed: max diff {max_diff:.2e} > tolerance {tolerance:.2e}"
        )

    log.info("✓ Sanity check passed — PyTorch and ONNX outputs match")


def export(
    config_path: str = "configs/config.yaml",
    checkpoint:  str = "checkpoints/best_model.pth",
) -> Path:
    """Full export pipeline: load → export → validate → sanity check."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = _get_device("cpu")  # export on CPU for maximum compatibility
    model  = build_model(config)
    load_checkpoint(Path(checkpoint), model)
    model  = model.to(device)

    onnx_path = Path(config["paths"]["onnx_model"])
    img_size  = config["data_gen"]["image_size"][0]

    export_to_onnx(
        model       = model,
        output_path = onnx_path,
        input_shape = (1, 3, img_size, img_size),
        opset       = config["export"]["opset_version"],
        dynamic_axes= config["export"]["dynamic_axes"],
    )

    validate_onnx(onnx_path)
    numerical_sanity_check(model, onnx_path)

    log.info(f"\n✓ ONNX model ready for production: {onnx_path.resolve()}")
    return onnx_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    args = parser.parse_args()
    export(args.config, args.checkpoint)
