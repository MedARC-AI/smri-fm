from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .generation import generate, validate_generation_inputs
from .manifest import write_accepted_manifest, write_manifest
from .publish import push_to_hf
from .qc import run_qc

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic MR brain data pipeline")
    parser.add_argument("--config", required=True, type=Path, help="YAML pipeline config.")
    parser.add_argument("--generator-repo", type=Path, help="Override generator_repo.")
    parser.add_argument("--output-dir", type=Path, help="Override output_dir.")
    parser.add_argument("--num-images", type=int, help="Override num_images.")
    parser.add_argument("--qc-threshold", type=float, help="Override qc.threshold.")
    parser.add_argument("--qc-metric", choices=["min", "mean"], help="Override qc.metric.")
    parser.add_argument(
        "--qc-mode",
        choices=["direct_synthseg", "preprocess_then_synthseg"],
        help="Override qc.mode.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate config without running generation or QC.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "synthetic_pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def main() -> None:
    args = parse_args()
    try:
        cfg = load_config(
            args.config,
            generator_repo=args.generator_repo,
            output_dir=args.output_dir,
            num_images=args.num_images,
            qc_threshold=args.qc_threshold,
            qc_metric=args.qc_metric,
            qc_mode=args.qc_mode,
        )
        setup_logging(cfg.output_dir)
        log = logging.getLogger("synthetic_pipeline")
        log.info("Loaded config: %s", args.config.resolve())
        log.info("Output dir: %s", cfg.output_dir)
        log.info("Generator repo: %s", cfg.generator_repo)

        if args.validate_only:
            validate_generation_inputs(cfg)
            log.info("Config validation succeeded.")
            return

        records = generate(cfg)
        manifest_path = cfg.output_dir / "manifest.csv"
        accepted_manifest_path = cfg.output_dir / "accepted_manifest.csv"
        write_manifest(manifest_path, records)

        records = run_qc(cfg, records)
        write_manifest(manifest_path, records)
        write_accepted_manifest(accepted_manifest_path, records)

        if cfg.push_to_hf.enabled:
            push_to_hf(accepted_manifest_path)

        accepted = sum(1 for record in records if record.get("qc_pass") is True)
        log.info("Pipeline complete: %d/%d image(s) accepted.", accepted, len(records))
    except Exception as e:
        logging.getLogger("synthetic_pipeline").error("%s", e, exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
