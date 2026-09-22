# S3a 完了確認・S3b 実行報告

- 実施日: 2026-09-16
- 対象: 修正済み作業ツリー（既存の未commit差分を保持）
- 判定: S3a は完了（PASS）。S3b は計算・診断出力のサブ範囲を完了（PASS）。

## S3a 完了確認

`leadlag.domain.distribution`を分布結果契約の正本として確認した。
`DistributionStatus`、`DistributionReason`、`DistributionAttempt`、`DistributionResolutionError`で、
sourceの結果、理由、試行順、horizonを追跡できる。`FallbackPolicy`はtyped reason/statusから
flat化時の監査失敗を決定し、alert文字列を検索して理由を昇格しない。

今回、on-demand計算の例外は常に`COMPUTATION_FAILED`として返し、provenance拒否は計算後の
`validate_distribution_provenance`の結果だけで`PROVENANCE_REJECTED`にする経路を確認した。
cache loader由来の人間向けalert分類はsource境界に限定し、fallback/decisionの安全判定には渡していない。
legacyの`is_available`、`is_flat`、`alerts`、`flat_decision`は移行中の互換出力として残っている。

## S3b 実施内容

### 完了した境界

- `src/leadlag/pipeline/gap_distribution.py`
  - `compute_gap_distribution()`を追加。
  - raw μ/Ω、gap補正、分母floor、gap μ/Ωを副作用なしで計算する。
  - gap生成のh=1/h=3/h=5が同じ関数を利用する。
  - h=1のStep 1 `Omega_struct`は明示引数として受け取り、on-demand BLPX共分散と混同しない。
- `src/leadlag/pipeline/gap_reporting.py`
  - accumulatorからticker summary、gap/distribution/omegaの日次・long形式を構築する。
  - 既存の6つの診断CSV名を安定した出力境界として維持する。
- `tools/research/compute_gap_adjusted_distribution.py`
  - h=1/h=3/h=5の重複するgap補正代数を共通関数へ移した。
  - 診断CSVの組立・保存をreporting adapterへ移した。
  - S1cで導入済みの`build_blpx_model`と、既存のbundle/provenance/監査処理を維持した。
- `src/leadlag/models/v2/gap_io.py`
  - 本番のon-demand gap計算も同じ`compute_gap_distribution()`を呼ぶように切り替えた。
  - gap取得値・係数選択・snapshot/PIT入力の責務は既存のadapterに残した。

### 意図的に残した範囲

研究固有のplot生成、入力取得の組立、portfolio評価と`report.md`生成は、研究用依存と評価仕様を
本番計算境界へ持ち込まないため、現行research scriptに残した。これはS3b全体の完了ではなく、
次の小単位で分離する残件である。cache/bundle manifestの整理はS3cの対象とする。

## 検証結果

| 検証 | 結果 |
|---|---:|
| S3a・source・Stage A-C回帰 | 53 passed |
| gap distribution / S3b core / S3a / source | 41 passed |
| S2a/S2b/model factory/pipeline/gap I/O 回帰 | 51 passed |
| gap生成script `--self-test` | PASS |
| 全テスト（`scripts/run_tests_parallel.sh`） | **653 passed / 17 warnings** |
| F821 lint（同スクリプト内） | PASS |
| 変更対象 Ruff | PASS |
| `compileall`（src/tests/tools） | PASS |
| `git diff --check` | PASS |

全テストはunit 552、integration 87、research 14（長時間スモーク4を含む）に分割され、全8 shardが成功した。
実データを用いたgap batch、broker発注、cacheの本番切替は実施していない。

## 次の作業

1. S3b残り: research固有のplot/input orchestrationを小さいadapterへ分離し、数値同値を確認する。
2. S2c/S3c: 互換shimの利用者切替、bundle schema/manifest、SQLite/file cacheの版追跡を整理する。
3. Step 1 `Omega_struct`とon-demand BLPXのraw μ/Ω差は、同一入力・horizonで分解比較してから仕様を決める。

## 追記（2026-09-16）

上記のS3b残り（research固有の入力組立、portfolio評価、plot/report）を
`research/diagnostics/{gap_inputs,gap_portfolio,gap_outputs}.py`へ分離して完了した。
判定と検証結果は[S3b全体完了報告](../20260916_structural_improvement_s3b/report.md)を参照する。
