# BLPX residual target ML overlay diagnostic

判定: **残差target候補は既知期間診断で不採用。前向き採否gateは未判定で、本番artifact/configは変更していない。**

## 仮説と教師target

MLに既存BLPX予測そのものを再学習させず、既存の方向付き予測から残る誤差を学ばせる。日t、銘柄jの教師は `sign(score[t,j]) × realized_return[t,j] − scale × sign(score[t,j]) × mu_gap[t,j]`。既存学習labelの固定10bpsは復元し、これは配分予測専用で注文費用ゲートとは分離する。主比較は同じfoldで再学習した従来raw-target ML、ML無効BLPXは参照値。scaleは0.8/1.0/1.2の±20%感度。

## 固定した評価

- 既存の監査済み2026-09-27学習行を再利用。年ごとに2020〜2024のexpanding walk-forwardを実施し、評価年直前5取引日をpurge。特徴量・LightGBM設定・コスト計算は固定。
- 年・銘柄行は過去に閲覧済みであり、独立fresh OOSではない。20日block bootstrapは既存evaluatorと同じ1,000回・seed 42。DSRは当study内のraw-target ML+3候補だけを使う名目値で、ML無効BLPXは参照であり、過去の関連ML試行を完全には補正しない。
- 記録先: `reports/20260929_ml_overlay_blpx_residual/daily_paired.csv` と `var/experiments/registry.jsonl`。

## 結果

| BLPX residual scale | Net Sharpe | Raw-target ML Sharpe | ML-off Sharpe | Compounded net | Raw-target compounded net | Max DD | Mean daily paired net difference vs raw target | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 6.725 | 6.780 | 6.741 | 39477.25% | 57355.07% | -7.21% | -0.032538% | [-0.043618%, -0.022796%] |
| 1.0 | 6.706 | 6.780 | 6.741 | 37523.68% | 57355.07% | -7.11% | -0.036929% | [-0.049875%, -0.025853%] |
| 1.2 | 6.706 | 6.780 | 6.741 | 36987.80% | 57355.07% | -7.10% | -0.038186% | [-0.052164%, -0.026247%] |

名目within-study DSR: 1.0。これはSharpeの帰無仮説に対する名目値でraw targetより優れる確率ではない。試行分散と探索履歴が不足するため、採用判断用DSRではない。

歴史診断の判定根拠: 残差target 3候補すべてでraw targetよりnet Sharpeが低く、paired 95%平均net差区間の上限も0未満。

## 判定と制約

この実装は相対配分向けの教師targetを追加したもので、MLに取引確率や総gross制御を持たせない。歴史provider `available_at`は未証明、09:10の凍結quote・完全な実約定費用・実口座建玉も揃っていないため、結果が良くても本番候補の有効性は未確定。現在の2025-12-30 cutoff artifact以降の完全な前向き250日paired labelsは0日であり、本targetのproduction training/artifact publicationも行っていない。

次段階は、同一artifact・同一入力の前向きpaired記録を蓄積し、必要な口座・注文cost正本が揃った後にincremental trade-value gateを別実験する。
