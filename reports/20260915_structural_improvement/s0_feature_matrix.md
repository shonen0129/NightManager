# S0機能基準マトリクス

これは構造変更後に同じ結果を比較するための基準ケース一覧です。値を推測して補完せず、修正後テストとmanifestで確認できる分岐を記録しています。

| ケース | 固定入力・経路 | S0で固定する期待 |
|---|---|---|
| h=1 cache | `tests/regression/baselines/matrices/mu_gap_20260814.npy` / `omega_gap_20260814.npy`、対応metadata | μ・Ω・signal date・horizon=1を一組で読む。来歴不正なら通常decisionへ進めない |
| h=3 / h=5 | 同ディレクトリの`*_h3_*` / `*_h5_*` bundle | horizon別入力・cache identityを混ぜない。追加horizon不正は除外またはflatの既存規則を維持 |
| cache欠損 | 固定`df_exec_20260814`＋9:10価格 | 許可されたon-demandを試し、入力不足・失敗時はflat。前日のbundleを流用しない |
| provenance不正 | `test_stage_abc_fixes.py`の未来signal date / metadata欠損ケース | on-demandまたは監査失敗flatへ進み、未検証分布からweightsを作らない |
| ML有効 | 解決済みproduction YAML、`models/ml_order_overlay/phase2_8` | legacy root artifactはloaderが拒否。再学習・versioned公開まで本番ML有効の再現性を完了扱いにしない |
| risk stop | broker fixture、既存建玉＋flat target | 新規増加・反転は禁止。確認済み在庫の削減と未完了状態の追跡は維持 |
| 注文部分失敗 | broker fixture（拒否、部分約定、取消後約定、照会失敗） | accepted / filled / partial / failed / pendingを分け、応答件数を成功件数にしない |
| 初日損失・flat | 固定PnL unit cases | 初期wealth=1を含む最大DD、持越し決済slippage、flat日を含む指標を維持 |
| 入力訂正 | DataFrame fingerprint / SQLite bundle fixture | 値・target・列identityの訂正でcacheを再利用しない。取得済みsnapshotは実行中に変わらない |

## 実行済み範囲

S0ではこのマトリクスを比較対象として固定し、既存の回帰・単体・統合テストで安全境界を確認した。各ケースの数値を新構造と比較する作業はS1以降で行う。全ケースがflatになったことだけでは合格としない。

出典: [s0_baseline_manifest.json](s0_baseline_manifest.json)、[s0_execution.json](s0_execution.json)、[第8回レビュー](../20260915_stage_abc_round8_review/review.md)。
