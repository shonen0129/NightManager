#!/usr/bin/env python
"""Deployment and Backup Tool for Production Residual-BLPX Configuration.

Safely replaces configs/production.yaml after verifying safety audits,
creating backups in configs/archive/ and generating unified diff patches.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src/ to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Apply Residual-BLPX Production Config")
    parser.add_argument("--source-config", default="configs/production/production.yaml", help="Current production config path")
    parser.add_argument("--target-config", default="configs/production/production_residual_blpx.yaml", help="New candidate config path")
    parser.add_argument("--backup-dir", default="configs/archive", help="Backup directory")
    parser.add_argument("--output-dir", default="results/production_residual_blpx_validation", help="Output directory of validation")
    parser.add_argument("--require-audit-pass", default="true", choices=["true", "false"], help="Enforce audit check pass before applying")
    parser.add_argument("--apply", action="store_true", help="Apply modifications (dry-run if omitted)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    src_path = ROOT / args.source_config
    target_path = ROOT / args.target_config
    backup_dir = ROOT / args.backup_dir
    out_dir = Path(args.output_dir) if args.output_dir.startswith("results") else ROOT / args.output_dir
    
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Verification of Audit status
    if args.require_audit_pass == "true":
        audit_file = out_dir / "audit.json"
        if not audit_file.exists():
            logger.error(f"Audit file {audit_file} not found. Please run validation script first.")
            sys.exit(1)
        with open(audit_file) as f:
            audit_res = json.load(f)
        if not audit_res.get("all_passed", False):
            logger.error("Compliance safety audits failed! Applying production change aborted.")
            sys.exit(1)
        logger.info("Safety audit check passed.")
        
    if not src_path.exists():
        logger.error(f"Source configuration path {src_path} does not exist.")
        sys.exit(1)
    if not target_path.exists():
        logger.error(f"Target configuration candidate {target_path} does not exist.")
        sys.exit(1)
        
    # 2. Read contents for diffing
    with open(src_path) as f:
        src_lines = f.readlines()
    with open(target_path) as f:
        target_lines = f.readlines()
        
    # Generate unified diff patch
    diff = list(difflib.unified_diff(
        src_lines, target_lines,
        fromfile=args.source_config,
        tofile=args.target_config
    ))
    
    diff_text = "".join(diff)
    diff_patch_path = out_dir / "production_config_diff.patch"
    with open(diff_patch_path, "w") as f:
        f.write(diff_text)
    logger.info(f"Configuration patch diff saved to: {diff_patch_path}")
    
    # 3. Apply changes (or perform dry-run)
    date_str = datetime.now().strftime("%Y%m%d")
    backup_filename = f"production_before_residual_blpx_{date_str}.yaml"
    backup_path = backup_dir / backup_filename
    
    if args.apply:
        logger.info(f"Backing up current configuration to: {backup_path}")
        shutil.copy(src_path, backup_path)
        
        logger.info(f"Overwriting {src_path} with {target_path}")
        shutil.copy(target_path, src_path)
        logger.info("Configuration deployed successfully.")
    else:
        logger.info("=== DRY-RUN MODE ===")
        logger.info(f"Would backup current config to: {backup_path}")
        logger.info(f"Would overwrite {src_path} with {target_path}")
        logger.info("Differences:")
        print(diff_text)
        logger.info("====================")


if __name__ == "__main__":
    main()
