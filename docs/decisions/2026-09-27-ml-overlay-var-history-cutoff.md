# ML overlay 2025年末カットオフと VaR 履歴再作成（2026-09-27）

## 決定

2025-12-31 までの情報に限定して作成した ML overlay artifact を登録し、train-end を守る日付別の artifact 選択で VaR の直近履歴を再構築した。2025-12-31 は東証休場日のため artifact の最終ラベル日は 2025-12-30。version は `20260926T192555935698Z-ee306a32f3ec`。

production `CURRENT` は今回の artifact を指す。切替前の active version `20260926T182011981905Z-f8b4ec178e88` は rollback 先として保持する。公開はユーザーの明示した再作成依頼と、それ以前に与えられた本番有効化の指示に基づく operator override であり、元の昇格基準を満たしたことを意味しない。

## 履歴とリスク

2025-07-29〜2026-09-25 の 269 営業日を作成した。2025-12-30 までは train-end 2024-12-20 の旧 version を使い、その翌営業日以降は今回の train-end 2025-12-30 version を使う。数値監査・リーク監査は全日 pass、fallback は 0 日。ADR 不足の 25 日は overlay を安全に skip した。

250 日 VaR99 は 3.053%（基準 3.00%）、ES99 は 4.591%（基準 4.00%）で、ともに停止基準を超えた。よって VaR データ不足による停止要因は解消したものの、実測リスクによる停止を維持する。250 日窓の tail 標本は 3 件なので、ES 値の不確実性も報告に残す。停止閾値や監査を緩めない。

学習期間後の結果は retrospective で独立 OOS ではない。新規 forward labels は 0 日、provider の historical `available_at` 来歴は未証明。数値昇格基準は不合格のまま。これらの条件が解消するまで、今回の履歴を性能優位性の根拠として扱わない。

## 実行経路の確認

本番設定から構成した ProductionRunner が新 version を読み込むことを確認し、2026-09-28 用の通常 VaR getter で 269 日の履歴が生成され、2 回目に cache hit することを確認した。入力にある暫定日付を除外し、直近完了営業日 2026-09-25 で終端した。broker adapter、decision 実行、注文は発生していない。

## 記録

詳細な方法・監査・コスト・制約は [実行レポート](../../reports/20260927_ml_overlay_var_history/report.md) を参照。機械可読の履歴、昇格ゲート、cache 検証は同レポート内の JSON、および production root の `HISTORY.json` / `promotions/20260927_var_history_cutoff_rebuild.json` に保存した。
