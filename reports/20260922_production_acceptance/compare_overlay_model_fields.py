from pathlib import Path
import pickle
for name,path in [('old',Path('models/ml_order_overlay/phase2_8/model.pkl')),('new',Path('models/ml_order_overlay/production_20260923/versions')/Path('models/ml_order_overlay/production_20260923/CURRENT').read_text().strip()/'model.pkl')]:
 m=pickle.loads(path.read_bytes())
 print('---',name,'---')
 print(sorted(m.__dict__.keys()))
 for k,v in m.__dict__.items():
  if k in {'model','regressor','estimator','lgbm_model','booster'}:
   print(k,type(v).__name__)
   for attr in ('get_params','num_trees','feature_name_'):
    if hasattr(v,attr):
     try:
      x=getattr(v,attr)()
     except TypeError: x=getattr(v,attr)
     print(attr, x if attr!='feature_name_' else {'count':len(x)})
