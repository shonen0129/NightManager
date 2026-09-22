# S1c / S1d 実行記録

- 実施日: 2026-09-15
- 対象: 修正済み作業ツリー（未commit差分を保持）
- 判定: **S1c PASS、S1d PASS（設定・構築の撤去範囲）**

## S1c 完了確認

S1cで切り替えたgap生成のh=1/h=3/h=5 BLPX構築を再確認した。

- `tools/research/compute_gap_adjusted_distribution.py` は
  `runner.model_factory.build_blpx_model`を使用する。
- gap生成はML overlayを適用しないため、BLPX単体factoryを選ぶ。
- `run_config.json`へ解決済みV2設定と`model_config_hash`（正規化JSONのSHA-256）を保存する。
- Step 1共分散、gap補正、診断、保存の数値経路は変更していない。

これにより、S1cの受入条件である「設定からBLPXを構築する境界の統一」と
「有効設定の再現情報の記録」を満たす。Step 1共分散とV2 on-demandのμ/Ω同値性、
provenance付きML artifact再生成、本番/BT weights照合はS3または運用残件として継続する。

## S1d 実行内容

次の旧経路を削除し、利用者を正本へ向けた。

| 撤去対象 | 移行先 |
|---|---|
| `execution.config`内の`_deep_merge`、`_load_yaml_with_base`、`_resolve_config_path` | `config.loader` |
| model moduleからのparser再export・下流の旧parser import | `config.schemas.parse_run_config` |
| `pipeline/compute_omega_struct.py`の重複BLPX factory | `runner.model_factory.build_blpx_model` |
| execution submoduleの`StrategyConfig`再export経由import | `config.schemas.StrategyConfig` |
| Stage ABC follow-up testsの旧module monkeypatch先 | 各正本loader |
| Runner内の旧`hasattr`設定探索 | validated `AppConfig.v2` と共通factory |

`execution.config.load_config_from_yaml`は、YAML合成だけでなく環境変数・broker設定を含む
`AppConfig`構築の正規アプリ境界であるため残した。これを経由する利用者を移すだけの
新wrapperは追加していない。`generate_v2_production_portfolio`のdict入力、
`data/cache.py` shim、`models/blpx.py`再exportは、利用者・保存形式を確認してからS2/S3で
撤去する別段階の対象である。

## 検証

| 検証 | 結果 |
|---|---|
| 設定loader、factory、Runner/BT、設定検証、execution submodule、Stage ABC回帰、V2 integration | **90 passed** |
| `execution.config` private alias不在テスト | PASS（上記に含む） |
| private旧関数・旧importの静的検索 | 対象参照なし（一般的な`hasattr`利用は除外） |
| Ruff、`git diff --check` | PASS |
| 変更対象 compileall | PASS |
| gap生成script self-test | PASS |
| `tests/` 全体（外側process timeout 1200秒、10 worker） | **661 passed / 17 warnings / 712.47秒** |
| import-linter（4契約） | **4 kept / 0 broken** |
| 対象 mypy（15 source files） | PASS |

全体Ruffは既存の研究スクリプトにある65件のlint負債でFAILした。S1c/S1dで変更した
ファイル群のRuffはPASSしており、既存負債を今回の完了条件へ持ち込んでいない。

S1dは既存の挙動を変える設定仕様変更ではなく、同じ正規化・構築を一つの入口へ寄せる
撤去である。全テストは外側process timeout付きで完走し、既存のStage A-C安全境界を
含む回帰を維持した。warningsは既存の数値計算由来で、失敗はない。
