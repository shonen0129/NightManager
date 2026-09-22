from pathlib import Path

from leadlag.execution.config import load_config_from_yaml
from leadlag.runner.model_factory import build_v2_model_bundle, resolve_overlay_settings

app = load_config_from_yaml(Path('configs/production/production.yaml'), strict=True)
enabled, path = resolve_overlay_settings(app)
bundle = build_v2_model_bundle(app, clear_blpx_cache=True)
print('ml_overlay_enabled', enabled)
print('resolved_overlay_path', path)
print('bundle_overlay_path', bundle.overlay_path)
print('artifact_version', bundle.overlay_model.metadata['artifact_version'] if bundle.overlay_model else None)
print('metadata_status', bundle.overlay_model.metadata['metadata_status'] if bundle.overlay_model else None)
