from pathlib import Path
import pickle
for name,path in [('old',Path('models/ml_order_overlay/phase2_8/model.pkl')),('new',Path('models/ml_order_overlay/production_20260923/versions')/Path('models/ml_order_overlay/production_20260923/CURRENT').read_text().strip()/'model.pkl')]:
 m=pickle.loads(path.read_bytes())
 print('---',name,'---')
 print('target_std',m.target_std)
 print('lgbm_type',type(m.lgbm).__name__)
 if hasattr(m.lgbm,'get_params'):
  print('lgbm_params',m.lgbm.get_params())
