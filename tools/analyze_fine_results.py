import pandas as pd
from pathlib import Path
p = Path('results') / 'fine_grid_local' / 'fine_grid_results.csv'
if not p.exists():
    print('Results file not found:', p)
    raise SystemExit(1)

df = pd.read_csv(p)
# Normalize presence of error column
if 'error' in df.columns:
    df_ok = df[df['error'].isna()].copy()
else:
    df_ok = df.copy()

for col in ['AR','R/R','RISK','MDD','Total Return']:
    if col in df_ok.columns:
        df_ok[col] = pd.to_numeric(df_ok[col], errors='coerce')

for label in df_ok['label'].unique():
    sub = df_ok[df_ok['label'] == label].copy()
    best_ar = sub.loc[sub['AR'].idxmax()]
    best_rr = sub.loc[sub['R/R'].idxmax()]
    print('\nSummary for', label)
    print('Best by AR: coef=', best_ar['topix_beta_coef'], 'bw=', best_ar['beta_window'], 'AR=', best_ar['AR'], 'R/R=', best_ar['R/R'], 'MDD=', best_ar['MDD'])
    print('Best by R/R: coef=', best_rr['topix_beta_coef'], 'bw=', best_rr['beta_window'], 'AR=', best_rr['AR'], 'R/R=', best_rr['R/R'], 'MDD=', best_rr['MDD'])

# Save condensed CSV
out = Path('results') / 'fine_grid_local' / 'best_summary.csv'
rows = []
for label in df_ok['label'].unique():
    sub = df_ok[df_ok['label'] == label].copy()
    best_ar = sub.loc[sub['AR'].idxmax()]
    best_rr = sub.loc[sub['R/R'].idxmax()]
    rows.append({
        'label': label,
        'ar_coef': best_ar['topix_beta_coef'],
        'ar_bw': best_ar['beta_window'],
        'ar_AR': best_ar['AR'],
        'ar_RR': best_ar['R/R'],
        'ar_MDD': best_ar['MDD'],
        'rr_coef': best_rr['topix_beta_coef'],
        'rr_bw': best_rr['beta_window'],
        'rr_AR': best_rr['AR'],
        'rr_RR': best_rr['R/R'],
        'rr_MDD': best_rr['MDD'],
    })

pd.DataFrame(rows).to_csv(out, index=False)
print('\nSaved best summary to', out)
