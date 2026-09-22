from pathlib import Path
import pickle

roots = {
    'old_phase2_8': Path('models/ml_order_overlay/phase2_8/model.pkl'),
    'new_production_20260923': Path('models/ml_order_overlay/production_20260923/versions') / Path('models/ml_order_overlay/production_20260923/CURRENT').read_text().strip() / 'model.pkl',
}
for name, path in roots.items():
    model = pickle.loads(path.read_bytes())
    print(f'--- {name} ---')
    print('class', type(model).__name__)
    metadata = getattr(model, 'metadata', {})
    print('metadata_keys', sorted(metadata) if isinstance(metadata, dict) else type(metadata).__name__)
    if isinstance(metadata, dict):
        for key in ('metadata_status','train_start','train_end','label_asof_end','artifact_version','model_sha256','target_std','target_type','seed','training_rows','training_dates','purged_trading_days','config_hash','data_hash','source_code_hash','lgbm_kwargs'):
            if key in metadata:
                print(key, metadata[key])
    for key in ('lgbm_kwargs','feature_names','cont_cols','use_ticker','use_classification','per_ticker_interactions','n_tickers','p_trade_scale'):
        if hasattr(model, key):
            value = getattr(model, key)
            if key in ('feature_names','cont_cols') and isinstance(value, (list, tuple)):
                value = {'count': len(value), 'head': list(value[:5]), 'tail': list(value[-3:])}
            print(key, value)
