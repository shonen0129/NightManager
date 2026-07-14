# A3 PIT Isotonic Diagnostic Report

## Data: 2481 aligned observations

## Spearman corr(pit_pct, next_return) = 0.1607 (p=0.0000)


### Tertile Sharpe Contributions

 bin   n  mean_ret  sharpe_contrib
 Low 833  0.003434        6.629929
 Mid 817  0.005295        8.293692
High 831  0.007161        9.603744

### Decile Analysis

 decile   n  mean_pit_pct  mean_next_return  std_next_return  sharpe_contrib
      1 214      0.000000          0.003484         0.008028        6.792521
      2 268      0.835110          0.003330         0.007255        7.183410
      3 262      4.110626          0.003451         0.008485        6.365860
      4 248     13.373656          0.004168         0.009205        7.087856
      5 247     32.979886          0.004349         0.008327        8.175301
      6 246     60.938508          0.006205         0.010394        9.343881
      7 250     80.395997          0.006491         0.011628        8.738239
      8 239     91.802276          0.006446         0.011595        8.701389
      9 220     97.790582          0.006075         0.009637        9.867228
     10 287     99.759572          0.008577         0.013319       10.079645

### Verdict: SUPPORTED

Low tertile has lower Sharpe contribution → 0.75x multiplier is justified.

Current 3-tertile RuleD is reasonable. No change needed.
