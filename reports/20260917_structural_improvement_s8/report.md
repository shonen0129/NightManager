# S8 実施報告 — CI・依存関係・文書境界

実施日: 2026-09-17

## 実装

- `.github/workflows/ci.yml`を追加した。Python 3.12、`uv sync --locked`、lock check、compileall、
  Ruff、mypy、import-linter、文書リンク、production wheel検査、`tests/`全体を一つのCIゲートへ組み込んだ。
  S0の比較manifest（コード版・入力fingerprint・回帰基準）もCI artifactとして保存する。
- `scripts/ci/verify_wheel.py`を追加し、wheelに`leadlag/`が含まれ、`research/`が含まれないことを検査する。
- `scripts/ci/validate_docs.py`を追加し、ARCHITECTURE・CI・scheduler手順・ADR・構造改善planの相対リンクを検査する。
- `pyproject.toml`へ次のimport契約を追加した。
  - production packageからresearch packageへの依存禁止
  - 純粋なgap/macro計算からI/O adapterへの依存禁止
  - domainからapplication configへの依存禁止
- `PortfolioDecision`からconfig schemaのruntime importを除去し、domain層を上位設定層から切り離した。
- Ruffのtargetを実行環境と同じ`py312`へ揃え、timeout utilityを型パラメータ構文へ更新して新しいlintゲートを通した。
- 同名の`src/leadlag/models/blpx.py`と`models/blpx/`について、package import・動的load・配布参照を確認し、
  重複root moduleを撤去した。正本は`models/blpx/` packageである。
- ARCHITECTURE、scheduler手順、refactor roadmap、構造改善planへ現在のV2/BLPX/CI境界と残件を反映した。

## ローカル検証

| 検査 | 結果 |
|---|---|
| `uv lock --check`（`UV_CACHE_DIR=/private/tmp/leadlag-uv-cache`） | PASS（123 packages resolved） |
| `compileall src/leadlag tests tools scripts src/research` | PASS |
| `ruff check src/leadlag tests tools/production tools/validation scripts/ci` | PASS |
| `mypy --config-file pyproject.toml src/leadlag` | PASS（141 files） |
| `lint-imports` | PASS（7 contracts kept / 0 broken） |
| 文書相対リンク | PASS（77 links） |
| wheel検査スクリプト | PASS（production-only fixture） |
| BLPX package import | PASS（`leadlag.models.blpx`はpackageを解決） |

`src/research/`全域のRuffには既存の負債があるため、CIではproductionと保守対象の学習入口を
厳格に検査する。既存負債をproductionコードの免除には使わない。実wheelのビルドは
`uv run --locked --with "build>=1.2,<2" python -m build --wheel`で試行したが、build-systemの
`wheel`取得時にDNS制限で失敗した（exit 1）。CIではbuildを解決できる環境でwheel生成・内容検査・
artifact uploadまで行う。

## 全体回帰

S8のコード変更（domain annotation、import契約、重複module撤去）後に、`tests/`全体を外側1800秒の
プロセス期限付きで実行した。

- `722 passed / 17 warnings`
- 実行時間: 715.43秒
- broker API、実口座、実scheduler登録、実データのML再学習は実行していない。

## 実装漏れと残件

- 完了: CI/static gates、document link gate、wheel package boundary、BLPX duplicate root module removal。
- S2c残件: `data/cache.py` compatibility shim、旧`df_exec`/snapshot入口、macro/ADRの厳密なavailable_at
  証拠。研究・外部利用者の切替と入力契約の証拠が必要なため、S8で無理に削除していない。
- S3c残件: VaR snapshot/worker lifecycle全体の分割とlegacy reader撤去。期限部品とcache identityは実装済みだが、
  実データ同値性と運用期限の検証が必要。
- S7残件: provenance付き実artifact再生成、本番/BT weights照合、実データ学習の性能記録。

残件は構造改善planの撤去条件付きで管理し、CI追加をもってS2c/S3c/S7の運用証拠が完了したとは扱わない。
