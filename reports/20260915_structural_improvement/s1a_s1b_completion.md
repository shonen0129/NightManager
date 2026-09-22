# S1a / S1b 実施報告

- 実施日: 2026-09-15
- 対象: 修正済み作業ツリー（未commit差分を保持）
- 判定: S1a完了、S1b完了（各段階の範囲内）

## S1a — 設定loader抽出

`src/leadlag/config/loader.py`へ、YAMLの`__base__`解決、相対パス解決、再帰deep merge、循環参照検出を抽出した。`src/leadlag/execution/config.py`の既存private入口は、S1dで利用者を切り替えて撤去するまでの互換aliasとして残している。

確認した契約:

- deep mergeはbase/overrideを変更しない
- 相対パスの継承を再帰的に解決する
- 循環`__base__`を構築前に拒否する
- 抽出loaderと既存execution入口の結果が一致する

## S1b — 本番/BTのモデル構築統一

`src/leadlag/runner/model_factory.py`を追加し、`V2ModelBundle`と`build_v2_model_bundle`を正本の構築境界とした。factoryはvalidated `AppConfig`だけを受け、同じ`ProductionV2RunConfig`からBLPXとV2 decision modelを構築する。overlayは設定有効時に一度だけロードし、明示されたartifact directoryは設定無効時も比較用overrideとしてロードできる。相対artifact pathはproject root基準で解決する。

`ProductionRunner`と`BacktestEngine._generate_v2_weights`をfactory利用へ切り替えた。BTの明示overlay object/path、並列worker時のBLPX cache clear、overlay enabled flagは保持している。VaRとgap生成は期限・共分散経路の差があるため、S1c以降で切り替える。

## 検証

| 検証 | 結果 |
|---|---|
| S1a/S1b unit: config loader, config validation, runner config, execution submodules, model factory | **28 passed** / 0.70秒 |
| V2 integration + regression baseline | 50 passed |
| 変更対象 Ruff | PASS |
| 対象 mypy（config/loader, runner, backtester） | PASS |
| 対象 compileall | PASS |
| document validation（リンク52件、source hash46件、S0–S8） | PASS |
| 全テスト分割実行（`scripts/run_tests_parallel.sh`、8 shard） | **全shard PASS**（unit 528、integration 87、research 8、その他4、P5の2 deselected、warnings 17） |

分割実行は外側のプロセス全体タイムアウト付きで実施し、全shardが終了コード0となった。第8回レビュー記録の650 passedはS0時点の既存証拠として引き継ぎ、今回のshard別結果と混同しない。実口座API、ML再学習、長期BTは実施していない。現行`models/ml_order_overlay/phase2_8`はlegacy artifactのため、factoryを通じても安全側に拒否される。

## 残件

- S1c: gap生成のBLPX構築を共通factoryへ切り替え、Step 1共分散とon-demandの差を分解する
- S1d: `execution.config` alias、下流の旧import、Runnerの互換分岐、重複factoryを利用者移行後に撤去する
- 本番ML artifactをprovenance付きで再生成し、同一artifact・同一入力の本番/BT weightsを照合する

上記S1c/S1dのうち構築切替と旧経路撤去は、後続の
[S1c/S1d実行記録](s1c_s1d_execution.md)で完了を確認した。VaRの期限付き準備とML
artifactの再生成・weights照合は引き続き未実施である。
