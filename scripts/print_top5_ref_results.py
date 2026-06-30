import pandas as pd
df = pd.read_csv('artifacts/novel_alpha/top5_reference_methods.csv')
methods = ['zscore_252','momentum_5d','rank_252','conditional_20pct','zscore_63','multiplicative']
print('='*105)
print('TOP5 ALTERNATIVE REFERENCE METHODS: Delta Sharpe Statistics')
print('='*105)
print(f"{'Method':<25} {'Mean':<10} {'Median':<10} {'Std':<10} {'WinRate':<10} {'Max':<10} {'Min':<10} {'MaxStreak':<10}")
print('-'*95)
for mn in methods:
    d = df[f'{mn}_delta'].dropna()
    wr = (d > 0).sum() / len(d) * 100
    max_s = 0; cur = 0
    for v in d.values:
        if v > 0: cur += 1; max_s = max(max_s, cur)
        else: cur = 0
    print(f"{mn:<25} {d.mean():<+10.4f} {d.median():<+10.4f} {d.std():<10.4f} {wr:<10.0f} {d.max():<+10.2f} {d.min():<+10.2f} {max_s:<10}")
df['year'] = df['start'].str[:4]
print()
print('='*105)
print('YEARLY MEAN DELTA SHARPE')
print('='*105)
hdr = f"{'Year':<6}"
for mn in methods: hdr += f' {mn[:18]:<20}'
print(hdr)
print('-'*130)
for yr, g in df.groupby('year'):
    line = f'{yr:<6}'
    for mn in methods:
        d = g[f'{mn}_delta'].dropna()
        line += f' {d.mean():<+20.4f}'
    print(line)
print()
print('='*105)
print('YEARLY WIN RATE (%)')
print('='*105)
hdr = f"{'Year':<6}"
for mn in methods: hdr += f' {mn[:18]:<20}'
print(hdr)
print('-'*130)
for yr, g in df.groupby('year'):
    line = f'{yr:<6}'
    for mn in methods:
        d = g[f'{mn}_delta'].dropna()
        wr = (d > 0).sum() / len(d) * 100 if len(d) > 0 else 0
        line += f' {wr:<20.0f}'
    print(line)
