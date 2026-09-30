# Version-aware ML overlay 269日 paired replay

実施日: 2026-09-29
判定: **回顧的なモデル費用ベース比較。前向き250日gateの判定には使用しない**

## 比較設計

- 期間: 2025-07-29〜2026-09-25、269営業日。全日を同一日付pairedで比較。
- ML有効: 2025-12-30まではartifact `20260922T011723828589Z-7e1a81ab2617`（train end 2024-12-20）、2025-12-31以降はartifact `20260926T192555935698Z-ee306a32f3ec`（train end 2025-12-30）。各予測日のtrain endより後のartifactだけを使用。
- ML無効: 同一 `DecisionInputs`、gap snapshot、cost/configから `ml_overlay_enabled` だけをfalseにした対照。
- 片道slippage 5.0bps、financing / borrow / reverseを含む本番モデル費用。実約定・口座PnLではない。
- 年率Sharpeは252日換算。95%区間は日次net差の20日non-circular moving-block bootstrap、5,000回、seed 20260924。

## 全269日

| 指標 | ML有効 | ML無効 | 差 (有効−無効) |
|---|---:|---:|---:|
| 複利net return | 218.7577% | 215.5599% | +3.1978% |
| 年率net Sharpe | 4.307 | 4.364 | -0.057 |
| 最大DD | -33.4010% | -31.9408% | -1.4602% |
| 平均日次turnover | 1.3154 | 1.3036 | +0.0118 |
| fallback日 | 0 | 0 | 0 |
| gross cost合計 | 67.6038% | 67.4246% | +0.1792% |
| 日次net差の平均 | — | — | +0.004366% |
| 日次net差の95% block-bootstrap区間 | — | — | [-0.008305%, +0.016804%] |

### artifact区間別

| 区間 | 日数 | ML有効複利net | ML無効複利net | 差 | 平均日次net差 |
|---|---:|---:|---:|---:|---:|
| 旧artifact | 99 | 83.5800% | 81.3086% | +2.2714% | +0.012821% |
| 2025年末cutoff artifact | 170 | 73.6342% | 74.0457% | -0.4116% | -0.000558% |

## 監査と限界

- paired日数: 269。同日index、入力dataframe fingerprint、gap snapshot fingerprintをML有効/無効で共有。
- ML有効: 数値監査 269/269、リーク監査 269/269。ML無効: 数値監査 269/269、リーク監査 269/269。
- overlay適用 244日、ADR特徴不足等でskip 25日。両側ともfallback 0 / 0日。モデルweight制約は全日に適合。
- 今回のcutoff済みgap snapshot fingerprintは前回VaR再生記録の値と一致しない（今回 `b6ef93ac49bd3379`、前回 `c99ca4e8d8e0ad75`）。本比較のML有効/無効は今回の同一snapshotを共有する。ML有効再生と前回保存returnの最大日次差は 0。
- macro providerの過去時点 `available_at` は未証明。09:10 midpointは凍結実quoteではなく既存df_execのhistorical inputから作ったproxyで、約定・板・feeの完全照合もない。
- 期間・入力・モデル版を見た後の回顧replayであり、新規の独立OOSでも、事前登録した現行artifact固定の前向き250日gateでもない。研究判定は **PENDING**。本番設定・CURRENT pointerは変更していない。

## 再現

`.venv/bin/python3 src/research/scripts/experiments/evaluate_ml_overlay_versioned_paired_20260929.py`

成果物: `var/results/20260929_ml_overlay_paired_269/summary.json`、`daily_paired.csv`、`audit_rows.json`、`paired_returns.pkl`。
