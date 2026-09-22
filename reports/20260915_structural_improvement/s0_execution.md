# S0実行記録

実行日: 2026-09-15
Baseline: `20260915_structural_improvement_s0`
Manifest: [s0_baseline_manifest.json](s0_baseline_manifest.json)
機能基準: [s0_feature_matrix.md](s0_feature_matrix.md)

## 結果

S0はPASS。修正済みの未commit作業ツリーを構造改善前の比較基準として固定した。

- 全テスト: **650 passed / 17 warnings**
  - 既存並列スクリプト: 622 passed
  - 同スクリプトが対象外とする regression/features: 28 passed
- `compileall`: PASS
- 本番`src/leadlag` Ruff: PASS
- F821 undefined-name precheck: PASS
- `mypy src/leadlag`: PASS（122 files）
- `lint-imports`: PASS（4 contracts kept / 0 broken、156 files、385 dependencies）
- batch/test shell syntax: PASS
- `git diff --check`: PASS

並列テストは1800秒の外側watchdogで実行し、正常終了した。集計ログは[s0_tests_parallel.log](s0_tests_parallel.log)、分割ログは[s0_tests_p1.log](s0_tests_p1.log)〜[s0_tests_p8.log](s0_tests_p8.log)に保存した。集計とコマンドは[s0_execution.json](s0_execution.json)に固定した。

## 固定したもの

manifestには、HEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285`、開始時のtracked diff hash、[tracked差分patch](s0_tracked_worktree.patch)、解決済みproduction YAML、regressionのdf_exec・h=1/3/5 gap bundle、ADR特徴量、依存lock、ML artifact参照、live gap storeの観測hashを保存した。認証情報と実口座状態は含めていない。

本番設定が参照する`models/ml_order_overlay/phase2_8`は、loaderがlegacy root artifactとして拒否した。この結果を基準状態として固定し、provenance付きartifactの再生成・本番/BT weights照合を完了扱いにはしていない。

## S0の限界

これは入力をhashで識別する比較基準であり、live storeや外部ネットワーク応答の複製ではない。cacheの実運用時点での変動、実口座の建玉・約定、schedulerの登録状態は検証していない。構造改善の次段階では、manifestの同じ固定入力を使って変更前後を別コード版・別model instanceで比較する。

この記録の追加ファイル自体はbaseline manifest作成後に生成されている。manifestのscopeに従い、S0結果は比較対象コードへ混ぜず、構造改善の証拠として扱う。
