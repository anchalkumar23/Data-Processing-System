"""
scripts/run_pipeline.py
────────────────────────
End-to-end pipeline runner — executes all stages in sequence:

    1. Generate synthetic dataset      (data_gen/generate_dataset.py)
    2. Train the CNN model             (training/train.py)
    3. Evaluate on the test set        (training/evaluate.py)
    4. Export model to ONNX            (training/export_onnx.py)

After all stages complete, you can start the API server with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Run:
    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --skip-datagen  # if data already exists
    python -m scripts.run_pipeline --skip-training --checkpoint checkpoints/best_model.pth
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline_run.log"),
        ],
    )


def section(title: str) -> None:
    """Prints a formatted section header."""
    line = "─" * 55
    log.info(f"\n{line}")
    log.info(f"  {title}")
    log.info(f"{line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full defect detection pipeline")
    parser.add_argument("--config",         default="configs/config.yaml")
    parser.add_argument("--skip-datagen",   action="store_true", help="Skip dataset generation")
    parser.add_argument("--skip-training",  action="store_true", help="Skip model training")
    parser.add_argument("--skip-eval",      action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-export",    action="store_true", help="Skip ONNX export")
    parser.add_argument("--checkpoint",     default=None, help="Resume training from checkpoint")
    args = parser.parse_args()

    setup_logging()
    t_total = time.perf_counter()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log.info("=" * 55)
    log.info("  Automated Visual Data Processing System")
    log.info("  Industrial Defect Detection Pipeline")
    log.info("=" * 55)

    # ── Stage 1: Dataset Generation ─────────────────────────
    if not args.skip_datagen:
        section("Stage 1/4: Synthetic Dataset Generation")
        t0 = time.perf_counter()

        from data_gen.generate_dataset import main as gen_data
        gen_data(args.config)

        log.info(f"[Stage 1] Done in {time.perf_counter() - t0:.1f}s")
    else:
        log.info("[Stage 1] Skipped (--skip-datagen)")

    # ── Stage 2: Training ────────────────────────────────────
    if not args.skip_training:
        section("Stage 2/4: Model Training (ResNet-18, GPU-accelerated)")
        t0 = time.perf_counter()

        from training.train import train
        train(args.config, resume=args.checkpoint)

        log.info(f"[Stage 2] Done in {time.perf_counter() - t0:.1f}s")
    else:
        log.info("[Stage 2] Skipped (--skip-training)")

    # ── Stage 3: Evaluation ──────────────────────────────────
    if not args.skip_eval:
        section("Stage 3/4: Test Set Evaluation")
        checkpoint = args.checkpoint or config["paths"]["checkpoints"] + "/best_model.pth"

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            log.warning(f"Checkpoint not found at {ckpt_path} — skipping evaluation.")
        else:
            t0 = time.perf_counter()
            from training.evaluate import evaluate
            results = evaluate(args.config, str(ckpt_path))
            log.info(f"  Test Accuracy: {results['accuracy']*100:.2f}%")
            log.info(f"  ROC AUC:       {results['auc']:.4f}")
            log.info(f"[Stage 3] Done in {time.perf_counter() - t0:.1f}s")
    else:
        log.info("[Stage 3] Skipped (--skip-eval)")

    # ── Stage 4: ONNX Export ─────────────────────────────────
    if not args.skip_export:
        section("Stage 4/4: ONNX Export for Production Serving")
        checkpoint = config["paths"]["checkpoints"] + "/best_model.pth"

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            log.warning(f"Checkpoint not found at {ckpt_path} — skipping ONNX export.")
        else:
            t0 = time.perf_counter()
            from training.export_onnx import export
            onnx_path = export(args.config, str(ckpt_path))
            log.info(f"[Stage 4] Done in {time.perf_counter() - t0:.1f}s")
    else:
        log.info("[Stage 4] Skipped (--skip-export)")

    # ── Summary ──────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    log.info("\n" + "=" * 55)
    log.info(f"  ✓ Pipeline complete in {total_time:.1f}s")
    log.info("")
    log.info("  Next step — start the API server:")
    log.info("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    log.info("")
    log.info("  Dashboard: http://localhost:8000")
    log.info("  API Docs:  http://localhost:8000/docs")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
