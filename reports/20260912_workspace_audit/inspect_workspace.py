"""Read-only repository inventory and resolved non-secret configuration."""
import ast
import importlib.metadata as md
import json
from collections import Counter
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from leadlag.execution.config import load_config_from_yaml

cfg = load_config_from_yaml(str(ROOT / 'configs/production/production.yaml'), strict=True)
summary = {'python': sys.version, 'v2': cfg.v2.model_dump(mode='json'), 'risk': cfg.risk.model_dump(mode='json'), 'strategy_costs': {k: getattr(cfg.strategy, k, None) for k in ['side_leverage','max_gross_exposure','max_net_exposure','slippage_bps','overnight_alpha_long','overnight_alpha_short']}}
summary['packages'] = {}
for p in ['pytest','pytest-xdist','pytest-timeout','ruff','mypy','pandas','numpy','scipy','lightgbm','cvxpy','import-linter']:
    try: summary['packages'][p] = md.version(p)
    except md.PackageNotFoundError: summary['packages'][p] = None
summary['inventory'] = {}
for d in ['src/leadlag','src/research','tests','tools','scripts','configs','reports','docs','archive']:
    files = [p for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    py = [p for p in files if p.suffix == '.py']
    counts = [(len(p.read_text(errors='replace').splitlines()), str(p.relative_to(ROOT))) for p in py]
    summary['inventory'][d] = {'files': len(files), 'python_files': len(py), 'python_lines': sum(n for n,_ in counts), 'largest_python': sorted(counts,reverse=True)[:12]}
summary['artifact_files'] = {}
for d in ['models/ml_order_overlay/phase2_8','var/experiments','var/live/production_residual_blpx']:
    summary['artifact_files'][d] = [str(p.relative_to(ROOT)) for p in (ROOT/d).glob('*') if p.is_file()]
summary['sqlite'] = {}
for p in [ROOT/'var/market_data/df_exec.sqlite',ROOT/'var/market_data/decision_cache.sqlite', ROOT/'var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite']:
    entry={'exists':p.exists()}
    if p.exists():
        entry['size_bytes']=p.stat().st_size
        db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,timeout=5)
        entry['tables']=db.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
        db.close()
    summary['sqlite'][str(p.relative_to(ROOT))]=entry
print(json.dumps(summary,ensure_ascii=False,indent=2))
