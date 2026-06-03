import pandas as pd
from pathlib import Path
p = Path('results') / 'grid_search_yearly' / 'grid_results.csv'
df = pd.read_csv(p)

# Clean
if 'error' in df.columns:
    df_ok = df[df['error'].isna()].copy()
else:
    df_ok = df.copy()

# Ensure numeric
for col in ['AR','RISK','R/R','MDD','Total Return']:
    if col in df_ok.columns:
        df_ok[col] = pd.to_numeric(df_ok[col], errors='coerce')

# Best by AR and by R/R
best_ar = df_ok.loc[df_ok.groupby('year')['AR'].idxmax()].set_index('year')
best_rr = df_ok.loc[df_ok.groupby('year')['R/R'].idxmax()].set_index('year')

compare = pd.DataFrame(index=sorted(df_ok['year'].unique()))
compare['ar_coef'] = best_ar['topix_beta_coef']
compare['ar_bw'] = best_ar['beta_window']
compare['ar_AR'] = best_ar['AR']
compare['rr_coef'] = best_rr['topix_beta_coef']
compare['rr_bw'] = best_rr['beta_window']
compare['rr_RR'] = best_rr['R/R']
compare['same'] = (compare['ar_coef']==compare['rr_coef']) & (compare['ar_bw']==compare['rr_bw'])

print('Per-year comparison (AR-best vs R/R-best):')
print(compare[['ar_coef','ar_bw','ar_AR','rr_coef','rr_bw','rr_RR','same']].to_string())

count_same = compare['same'].sum()
count_total = len(compare)
print(f"\nYears with identical best settings by AR and R/R: {count_same}/{count_total}")

# Frequency of best coefficients by R/R
freq = best_rr['topix_beta_coef'].value_counts().sort_index()
print('\nFrequency of topix_beta_coef being best under R/R:')
print(freq.to_string())

# Save comparison
compare.reset_index().to_csv(Path('results') / 'grid_search_yearly' / 'compare_ar_vs_rr.csv', index=False)
print('\nSaved compare_ar_vs_rr.csv')
