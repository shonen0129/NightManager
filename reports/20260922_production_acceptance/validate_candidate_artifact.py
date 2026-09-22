from pathlib import Path

from leadlag.models.ml_overlay_artifact import load_overlay_model

root = Path('var/results/20260922_production_acceptance/artifacts/evaluation_2025')
model = load_overlay_model(root)
print('loaded', type(model).__name__)
print('artifact_version', model.metadata['artifact_version'])
print('metadata_status', model.metadata['metadata_status'])
print('model_sha256', model.metadata['model_sha256'])
