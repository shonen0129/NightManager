# 2025年末カットオフ artifact と VaR 履歴の再作成

## 結果

2025-12-31 までの教師ラベルを使って ML overlay artifact を再作成し、train-end を越えない version 別の VaR リターン履歴を作成した。2025-12-31 は東証取引日ではないため、実際の最終ラベル日は 2025-12-30 である。

artifact version は `20260926T192555935698Z-ee306a32f3ec`、SHA-256 は `c1a5b9354dae4b6507b41453c3b1c6a90451a81bcf0d3bc78b07939f421cb1e9`。学習範囲は 2015-01-05 から 2025-12-30、2,614 日、17 銘柄あたり 2,614 行、合計 44,438 行。既存の LightGBM パラメータを維持した。

履歴は 2025-07-29〜2026-09-25 の 269 営業日で構成した。2025-12-30 までは旧 artifact `20260922T011723828589Z-7e1a81ab2617`（99 日）、2025-12-31 以降は今回の artifact（170 日）を適用した。269 日のうち 244 日は overlay を適用し、25 日は ADR 入力が欠けていたため既存の安全な overlay skip とした。V2 fallback は 0 日。

## 監査と VaR 判定

全 269 日で数値監査・リーク監査が pass し、flat / failed は 0 日。モデルウェイトの最大 gross は 2.0、最大絶対 net は約 `3.75e-16`、実効 side leverage は 1.5 だった。

250 日の履歴 VaR99 は 3.053%（停止基準 3.00%）、ES99 は 4.591%（停止基準 4.00%）で、ともに基準を超過した。ES の tail 標本は 3 件に留まる。したがって履歴不足は解消したが、計測されたリスク超過による発注停止は継続する。閾値や停止動作は変更していない。

この retrospective replay は独立した OOS 検証ではない。再作成以降の前向き教師ラベルは 0 日で、学習データ provider の過去時点 `available_at` 来歴も未証明のため、数値昇格ゲートは不合格のまま記録した。ユーザー指示で production `CURRENT` は今回の version に切り替え、切替前の version `20260926T182011981905Z-f8b4ec178e88` を rollback 用に保持している。これはゲート合格や優位性の主張ではない。

## 実行時確認

2026-09-28 の production VaR 履歴取得経路を使い、履歴を作成して cache に保存後、同一キーの再取得が cache hit になることを確認した。初回生成 87.196 秒、cache hit 2.254 秒。入力スナップショットに 2026-09-28 の暫定行があったが、履歴終端は直近完了営業日 2026-09-25 に制限された。VaR は 250 標本を使用した。decision / broker 経路は実行せず、発注数は 0。

2026-09-25 終端のローカル検証済み macro snapshot を再利用した。外部データ取得は失敗していたため、source provider の過去時点可用性が確認できたことを意味しない。元の raw macro cache は書き換えていない。

累積コスト内訳は slippage 0.58197、financing 0.02891、borrow 0.00887、reverse 0.05629、合計 0.67604（履歴リターンと同じ率単位）。9:10 執行価格の近似を含む既存バックテスト上のコストであり、実約定との全件突合を示すものではない。

## 成果物

- 候補 artifact と学習内訳: [`2025_cutoff_candidate.json`](2025_cutoff_candidate.json)
- 履歴・監査の集計: [`history.json`](history.json)、[`history_audits.json`](history_audits.json)
- 公開・ゲート記録: [`publication.json`](publication.json)、[`20260927_var_history_cutoff_rebuild.json`](../../models/ml_order_overlay/production_20260923/promotions/20260927_var_history_cutoff_rebuild.json)
- 実行時 cache 検証: [`runtime_cache.json`](runtime_cache.json)
- 生成した日次リターン: [`versioned_var_returns.pkl`](../../var/results/20260927_ml_overlay_var_history_2025/history/versioned_var_returns.pkl)
- 本番履歴スケジュール: [`HISTORY.json`](../../models/ml_order_overlay/production_20260923/HISTORY.json)

再現用スクリプトは `src/research/scripts/experiments/retrain_ml_overlay_var_history_2025.py`、`create_var_history_2025_cutoff.py`、`publish_ml_overlay_var_history_2025.py`、`prewarm_var_history_20260928.py`。
