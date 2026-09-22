#!/bin/bash
set -euo pipefail
cd /Users/shonen/leadlag
exec env PYTHONPATH=src .venv/bin/python reports/20260922_production_acceptance/run_scheduler_like_dry_run.py
