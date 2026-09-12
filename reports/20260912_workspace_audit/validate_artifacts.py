"""Check audit references and JSON evidence without modifying source/runtime data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
errors: list[str] = []
references: set[str] = set()
manifest: list[dict] = []

for markdown in (AUDIT / 'report.md', AUDIT / 'README.md'):
    for target in re.findall(r'\]\((/Users/shonen/leadlag/[^)]+)\)', markdown.read_text()):
        references.add(target)
        match = re.fullmatch(r'(.*?)(?::(\d+))?', target)
        assert match
        path = Path(match.group(1))
        if not path.is_file():
            errors.append(f'missing reference: {target}')
        elif match.group(2) and int(match.group(2)) > len(path.read_text().splitlines()):
            errors.append(f'invalid line: {target}')

json_names = [
    'inventory.json', 'data_summary.json', 'check_summary.json',
    'reproductions.json', 'integration_reproductions.json',
    'model_probes.json', 'broker_contract_probes.json', 'data_contract_probes.json',
]

def check_no_probe_error(value: object, name: str) -> None:
    if isinstance(value, dict):
        if 'probe_error' in value:
            errors.append(f'probe failed in {name}: {value}')
        for item in value.values():
            check_no_probe_error(item, name)
    elif isinstance(value, list):
        for item in value:
            check_no_probe_error(item, name)

for name in json_names:
    try:
        check_no_probe_error(json.loads((AUDIT / name).read_text()), name)
    except (OSError, ValueError) as exc:
        errors.append(f'{name}: {exc}')

paths = {Path(re.sub(r':\d+$', '', ref)) for ref in references}
paths.update((ROOT / 'configs/production/production.yaml', ROOT / 'configs/base.yaml'))
for path in sorted(paths):
    if path.is_file() and not path.is_relative_to(AUDIT):
        manifest.append({
            'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })

revision = subprocess.run(
    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10,
).stdout.strip()
(AUDIT / 'source_manifest.json').write_text(json.dumps({
    'audit_date': '2026-09-12', 'git_revision': revision,
    'note': 'Existing user changes were present; hashes describe observed files.',
    'files': manifest,
}, ensure_ascii=False, indent=2) + '\n')

log = (AUDIT / 'pytest_full.log').read_text()
if '1 failed, 576 passed, 17 warnings' not in log:
    errors.append('full-test summary does not match report')

result = {
    'status': 'PASS' if not errors else 'FAILED',
    'local_references_checked': len(references),
    'evidence_json_checked': len(json_names),
    'source_files_hashed': len(manifest),
    'errors': errors,
    'note': 'Checks artifact structure, not whether production findings are fixed.',
}
(AUDIT / 'artifact_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
