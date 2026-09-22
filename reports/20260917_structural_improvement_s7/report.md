# S7a / S7b 実施報告

実施日: 2026-09-17

## 判定

S7a（ML学習・artifact互換）とS7b（配布分離・研究入口）を PASS とする。
本番の fitted class path と artifact の安全境界を維持したまま、学習・推論・特徴量・保存の責務を分離した。

## 実装

| 領域 | 正本 |
|---|---|
| pickle互換コンテナ | `src/leadlag/models/ml_order_overlay.py::MLOrderOverlayModel` |
| 特徴量・推論前処理・fit/inference共通列順 | `src/leadlag/models/ml_overlay_features.py` |
| artifact保存・読込・provenance | `src/leadlag/models/ml_overlay_artifact.py` |
| 本番overlay適用 | `src/leadlag/models/ml_overlay_inference.py` |
| 学習・学習データ収集・LightGBM fit | `src/research/experiments/ml_overlay_training.py` |
| 運用学習CLI | `tools/production/train_ml_order_overlay.py` |

旧 `leadlag.models.ml_order_overlay` は class と分割先の薄い再exportを提供する互換境界として残した。旧学習関数名は research への移行案内を返す fail-closed stub とし、production codeから research を動的importしない。artifactは引き続き version directory + `CURRENT` + SHA-256 + verified provenance を要求し、legacy root形式を拒否する。

`pyproject.toml` の setuptools 探索対象を `leadlag*` に限定し、研究依存は `research` extra と [研究環境手順](../../docs/RESEARCH_ENV.md) に分けた。LightGBMは既存artifactの本番推論に必要なため `nonlinear` extra に残した。新規学習artifactの `data_hash` は共有 `dataframe_fingerprint` を使い、ticker列順・dtype・index schemaの変更も検出する。学習イベントには解決済みrun config、入力hash、artifact versionを記録し、例外・KeyboardInterruptは`interrupted`として記録する。

## 検証

- ML overlay / factory / Stage ABC 回帰: 52 passed
- S7境界テスト: 7 passed（学習成功・中断のregistry観測、旧production学習入口のfail-closedを含む）
- compileall: `src/leadlag`, `src/research`, `tests`, `tools` PASS
- Ruff: 変更対象 PASS
- mypy: 変更対象5モジュール PASS
- import-linter: 4 contracts kept / 0 broken
- subprocess隔離import: `leadlag.models.ml_order_overlay` のimportで `research` 未ロードを確認
- setuptools `find_packages(where="src", include=["leadlag*"])`: 20 production packages、`research` 0件を確認
- `tools/production/train_ml_order_overlay.py --help` と production `leadlag --help`: 研究学習入口・本番CLIの起動を確認
- class path: `leadlag.models.ml_order_overlay.MLOrderOverlayModel` を確認
- 既存artifactの保存・読込・digest・provenance回帰: PASS
- 全体テスト: 722 passed（18 warnings）。S7境界7件を含む最終状態でPASS。

実wheelビルドは、環境に `wheel` がなく、ネットワーク制限によりbuild依存を取得できなかった。したがってwheelの実ファイル内容をPASSとは扱わず、setuptools設定の静的確認と隔離importで代替した。

## 実装漏れ・残件

今回のS7範囲で、コード境界の実装漏れは確認されなかった。次は次の運用・研究入力が必要である。

1. provenance付き本番ML artifactを実データから再生成する。
2. 同一artifact・同一入力で本番RunnerとBacktestの `scores_overlay` / `w_final` を照合する。
3. 実データの研究学習を実行し、成功・中断・指標を `ExperimentRegistry` へ記録する。
4. build依存を利用できるCIまたは配布環境でwheel内容を検査し、`research/` が含まれないことを確認する。

これらはデータ・CI・運用環境を要するため、今回ローカルで作成したコード分離のPASS判定とは分けて保留する。ML有効本番再現性や本番昇格を完了扱いにはしない。
