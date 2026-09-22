# S1c 実行記録 — gap生成のモデル構築統一

実行日: 2026-09-15
対象: `tools/research/compute_gap_adjusted_distribution.py`、`src/leadlag/runner/model_factory.py`
判定: **PASS（S1cの構築切替範囲）**

## 変更

gap生成のh=1モデルを、`ProductionBLPXModel(v2_cfg.blpx)`の直接構築から
`build_blpx_model(app_config)`へ切り替えた。multi-horizonのh=3/5も同じfactoryを使う。

gap生成はML order overlayを適用する入口ではないため、`build_v2_model_bundle`ではなく
BLPX単体factoryを呼ぶ。これにより、本番設定がML有効でも、gap生成がlegacy artifactの
読込に依存しない。factoryはvalidated `AppConfig`を受け、`app_config.v2.blpx`だけを
モデルへ渡す。Step 1の`Omega_struct`読込、gap補正、診断集計、保存形式は変更していない。

生成物の`run_config.json`に、次を追加した。

- 有効V2モデル設定のSHA-256 (`model_config_hash`)
- 秘密情報を含まない解決済みV2設定 (`effective_v2_config`)

設定hashはプロセス内のPython `hash()`ではなく、JSON正規化後のSHA-256で計算する。

## S1b完了確認

S1bの既存実装を再確認し、`ProductionRunner`と`BacktestEngine._generate_v2_weights`が
`build_v2_model_bundle`を使うこと、factoryがraw dictを拒否し、BLPX objectをV2 modelと
共有することを確認した。VaRはS1bの範囲外であり、gap生成と同様にS1cで切り替える対象には
含めず、期限付き準備を含む切替は後続作業とする。

## 検証

| 検証 | 結果 |
|---|---|
| S1b対象（factory、runner/config、V2 integration） | **66 passed** / 0.90秒 |
| S1c対象（factory + gap distribution unit） | **25 passed** / 11.28秒 |
| gap生成script self-test | **PASS** |
| `compileall`（factory、gap script、test） | PASS |
| `git diff --check` | PASS |

self-testは数値変換、PSD対称性、PIT境界、ticker順を確認する。実データを使うgap生成、
ネットワーク取得、ML再学習、実口座API呼出は行っていない。

## 数値同値性の範囲と残件

今回共通化したのは**設定からBLPX objectを構築する部分**である。Step 1の構造共分散を
使うgap生成と、V2 on-demandの`build_raw_distribution`は計算経路が異なるため、同一日の
μ/Ωの同値性を本変更だけで主張しない。S3bでprior、target、共分散版、gap補正係数を
分解比較する。

S1dで`execution.config` alias、下流の旧import、重複factoryを撤去した（詳細は
[S1c/S1d実行記録](s1c_s1d_execution.md)）。VaRの期限付き設定・overlay準備の共通化、
provenance付きML artifact再生成、本番/BT weights照合も残る。
