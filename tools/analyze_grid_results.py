import pandas as pd
from pathlib import Path
p = Path('results') / 'grid_search_yearly' / 'grid_results.csv'
df = pd.read_csv(p)

# Keep only successful rows
df_ok = df[df['error'].isna()] if 'error' in df.columns else df

# For each year find best by AR
best_by_year = df_ok.loc[df_ok.groupby('year')['AR'].idxmax()].sort_values('year')
print('Best topix_beta_coef per year by AR:')
print(best_by_year[['year','topix_beta_coef','beta_window','AR','RISK','R/R','MDD','Total Return']].to_string(index=False))

# Aggregate: how often each beta_coef is best
counts = best_by_year['topix_beta_coef'].value_counts().sort_index()
print('\nCount of best beta_coef across years:\n', counts.to_string())

# Save summary
best_by_year.to_csv(Path('results') / 'grid_search_yearly' / 'best_by_year.csv', index=False)
print('\nSaved best_by_year.csv')
