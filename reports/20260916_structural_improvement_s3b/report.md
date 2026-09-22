# S3b 構造改善 実施報告

## 判定

**S3b 全体: PASS（2026-09-16）**

gap 診断入口の責務を、入力組立・一日分の数値計算・保存・portfolio評価・診断出力へ分離した。
挙動を変えるモデル計算や本番設定の変更は行っていない。

## 実装

- `src/research/diagnostics/gap_inputs.py`
  - h=1 は既存の `_prepare_common_inputs(df_exec)` 呼出しを維持。
  - h=3/h=5 は、元の一日 frame から作った target override と horizon を明示して入力を組み立てる。
  - raw/preprocessed market data、TOPIX trade return、Tachibana realtime 注入を入力adapterへ集約する。
- `src/leadlag/pipeline/gap_distribution.py`
  - raw μ/Ω、gap補正、分母 floor、数値監査用の値を純粋な結果型で計算する。
- `src/leadlag/pipeline/gap_reporting.py`
  - accumulator から安定した6種類の診断 DataFrame/CSV を作成する。
- `src/research/diagnostics/gap_portfolio.py`
  - 実現 gross/net、片道 slippage、raw/gap IR、turnover、gross/net exposure を評価する。
- `src/research/diagnostics/gap_outputs.py`
  - lookahead-free ex-ante cost、PIT bin比較、監査JSON、plot、reportを研究専用出力として担当する。
- `tools/research/compute_gap_adjusted_distribution.py`
  - 上記境界を呼ぶ orchestration に整理し、production pathへ研究用描画依存を持ち込まない。

## 検証

- S3b境界テスト + 既存gapテスト: **29 passed**
- self-test (`--self-test`): **PASS**
- Ruff（新規モジュールとgap入口）: **PASS**
- compileall（対象モジュール）: **PASS**

- 全体テスト（unit/integration/research/features/regression の分割実行）: **693 passed**

全体実行は `scripts/run_tests_parallel.sh` を1800秒 watchdog付きで実行し、
同スクリプトが対象外とする `tests/features/` と `tests/regression/` も別途実行した。

## 残件

S3bの範囲外として、S2cの互換shim撤去条件とS3cのVaR lifecycle全体分割を残す。
実データを使ったcache/on-demand長期同値性、ML artifact再生成、本番昇格、実口座API操作は本報告の対象外である。
