# 防止策シミュレーション: 直近 5 日の影響

> 作成日: 2026-08-01
> すべての判定は過去情報のみを使用（ルックアヘッドなし）

## 概要

直近 5 日の baseline:
- Gross 累積: -601.84 bps
- Net 累積: -665.37 bps

## シナリオ

| シナリオ | パラメータ | Gross 5日累積 (bps) | Net 5日累積 (bps) | 直近5日間ヒット日数 |
|---|---|---:|---:|---:|
| baseline | alpha_long=0.75, alpha_short=0.50, no stop | -601.84 | -665.37 | 0 |
| alpha_long_0.5_short_0.25 | alpha_long=0.5, alpha_short=0.25 | -503.60 | -581.30 | 0 |
| alpha_long_0.0_short_0.0 | alpha_long=0.0, alpha_short=0.0 | -362.70 | -462.70 | 0 |
| daily_loss_stop_-0.5pct_scale0.0 | if prior net < -50 bps then scale=0.0 | -263.27 | -300.43 | 2 |
| daily_loss_stop_-1.0pct_scale0.0 | if prior net < -100 bps then scale=0.0 | -263.27 | -300.43 | 2 |
| daily_loss_stop_-1.5pct_scale0.0 | if prior net < -150 bps then scale=0.0 | -580.54 | -631.94 | 1 |
| daily_loss_stop_-2.0pct_scale0.0 | if prior net < -200 bps then scale=0.0 | -580.54 | -631.94 | 1 |
| daily_loss_stop_-0.5pct_scale0.5 | if prior net < -50 bps then scale=0.5 | -421.91 | -467.35 | 3 |
| daily_loss_stop_-1.0pct_scale0.5 | if prior net < -100 bps then scale=0.5 | -421.91 | -467.35 | 3 |
| daily_loss_stop_-1.5pct_scale0.5 | if prior net < -150 bps then scale=0.5 | -591.19 | -648.97 | 1 |
| daily_loss_stop_-2.0pct_scale0.5 | if prior net < -200 bps then scale=0.5 | -591.19 | -648.97 | 1 |
| drawdown_stop_3pct_scale0.0 | if DD < -3% then scale=0.0 | 0.00 | 0.00 | 5 |
| drawdown_stop_5pct_scale0.0 | if DD < -5% then scale=0.0 | 0.00 | 0.00 | 5 |
| drawdown_stop_3pct_scale0.5 | if DD < -3% then scale=0.5 | -436.18 | -487.68 | 2 |
| drawdown_stop_5pct_scale0.5 | if DD < -5% then scale=0.5 | -591.19 | -648.97 | 1 |
| ic20_stop_lt_0.0_scale0.0 | if 20-day IC < 0.0 then scale=0.0 | -601.84 | -665.37 | 0 |
| ic20_stop_lt_0.0_scale0.5 | if 20-day IC < 0.0 then scale=0.5 | -601.84 | -665.37 | 0 |
| ic20_stop_lt_-0.05_scale0.0 | if 20-day IC < -0.05 then scale=0.0 | -601.84 | -665.37 | 0 |

## 説明

- `alpha_long/short=0` はオーバーナイト持越しを完全に停止するケース。5日の Net 損失が最も減少するが、平日のコスト増と長期パフォーマンスへの影響は別途検証が必要。
- `daily_loss_stop` は前日 Net 損失が閾値を下回った翌日にポジションを縮小・停止する。最初の1日は防げない。
- `drawdown_stop` はアカウント資産の直近高値からの下落率で判定。閾値を -3% や -5% に設定すれば 7/30 以降に停止する。
- `ic20_stop` は過去20日の w vs JP target Spearman（実現した情報係数）が閾値を下回った日にポジションを縮小する。直近の悪化には 20 日平均では反応しにくい。

## 注意

- 本シミュレーションは直近 5 日のみを対象としており、長期パフォーマンス（Sharpe、DD、ターンオーバー）への影響は含まれていない。
- コストは `BacktestEngine` と同一の式で再計算している。
- `daily_loss_stop`・`drawdown_stop` はシミュレーション内で当日の損失が翌日の停止判定に影響するパス依存を考慮している。
