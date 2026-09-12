# AGENTS.md / Skill 整合レビュー

作成日: 2026-09-08

## 対象と結論

対象はリポジトリの `AGENTS.md` と `.agents/skills/` の9個。参照先の `docs/スタック再発防止策.md` にも同じ誤った復旧手順があったため修正した。システム・外部プラグインの Skill、本番コード・設定、作業中の subsector 資料は変更対象に含めていない。

共通規約を AGENTS.md、作業別手順を Skill、履歴と結果を ADR / reports / 不採用索引へ分けた。リーク禁止、2010–2014 の分離、本番ユニバース、モデルウェイトの net/gross 制約、当日 gap、監査維持、コード変更後の全体回帰、過学習補正・OOS・shadow の要件は保持した。文書のみの変更には文書固有の検証を適用する。

Skill の description は対象作業を明確にし、必要時に本文を読む構成とした。[公式 Skill ガイド](https://learn.chatgpt.com/docs/build-skills)と[AGENTS.md ガイド](https://learn.chatgpt.com/docs/agent-configuration/agents-md)も参照した。

## 発見事項と修正

| 問題 | 根拠・影響 | 修正 |
|---|---|---|
| 適用条件の重複 | モデル変更だけで実験・refactor・全体監査等が連鎖する description | 各 Skill の対象と依頼範囲を明示 |
| レビュー範囲の矛盾 | code-review の「全体」「修正禁止」が差分レビュー・修正依頼にも適用される | 差分/全体、レビューのみ/修正込みを分岐 |
| 不要な完了条件 | refactor が対象 Phase の未了を全てゼロにするよう指示 | 依頼範囲の完了と Phase 全体の完了を区別 |
| 旧実装への参照 | V1 `run_backtest()`、旧 `models/base.py`、存在しない `_check_syntax.py`、未提供検索ツール | V2、実在パス、標準検索へ更新 |
| config / pipeline の陳腐化 | `production.yaml` は現在 `configs/base.yaml` を継承。モデル内部は `models/v2/` に分割済み | 継承解決、分布 source chain、`var/` 正本を明記 |
| gap欠損の説明不足 | 旧テスト/境界/レポート Skill は cache 欠損即フラット扱い | cache → on-demand → flat と必要依存を区別 |
| 監査結果の過大解釈 | 日付監査だけで窓の非リークを証明、PSDを正定値と記述、数値/リーク失敗の処理を混同 | 実窓の摂動、PSD、下流の停止処理まで検証 |
| exposure の尺度不足 | `baseline_gross=2.0`、`side_leverage=1.5`、`risk.max_gross_exposure=3.0` が併存 | モデルウェイトと実効 exposure を区別。制限値は緩和しない |
| ハング対策自体の誤り | ThreadPoolExecutor の終了待ち、存在しない例外名、lock の存在と保持の混同 | 現行ラッパーの限界・プロセス全体の停止期限を明示 |
| 復旧による二重注文・排他破壊 | `kill -9`、lock 一括削除、API有効での無条件再実行 | 対象特定→TERM→必要時KILL、ロック保護、注文状態照合 |
| 実験条件の固定・内部矛盾 | 全区間正/負区間2個まで、8/12、2015–2026、61/5日、0.5 SE ≈ 0.01 が一律必須 | 対象期間・ラベル・試行に応じて事前設計。感度分析とDSRは維持 |
| 再現証跡の破棄 | 不採用ならコード/データ削除、AGENTS / Skill に成績追記 | 再現用コード/設定/参照を保持し、結果は reports / registry / 不採用索引へ |
| テストテンプレートの不足 | `pass` の骨組み、同一modelの2回呼出を変更前後比較にする例 | 最小再現と独立した版/インスタンスの比較へ |
| 全テストという呼称の不足 | 分割スクリプトは unit/integration/research のみ、features/regression が未収録 | `tests/` 全体を検証する手順を明記 |
| 指標集計と registry の注意不足 | helper はフラット日を除外し、trials を同名レコード数で上書き | 全評価日を主指標にし、必要なら直接 ExperimentRecord を記録 |

## 必要十分性の確認

| 作業 | 必要な規約・手順 |
|---|---|
| 文書/Skillの変更 | 構造・参照・意味の整合。戦略実験や全 Phase 完了へ拡大しない |
| 挙動維持の整理 | refactor → 必要な回帰 → 全体回帰 → 差分レビュー。OOS は性能比較時 |
| 性能変更の実験 | 仮説・比較条件・試行記録・境界/非リーク・V2・過学習補正・OOS・報告 |
| 本番昇格 | 研究採用と分離し、shadow・有効設定・運用手順を確認 |
| 障害診断 | 全体の期限、待機箇所・対象プロセス・注文状態を確認し、安全に復旧 |

## 検証

- skill-creator の `quick_validate.py` による9個の形式・命名検証: PASS。
- 改訂文書中の明示的なリポジトリパス52件: 存在確認 PASS。
- バックテスト結果キー: `backtester.py` の AST と照合 PASS。
- 本番設定の継承先2ファイルと fallback 既定値: 読み取りで確認。
- `git diff --check`: PASS。
- 検証は `.venv/bin/python` を使用し、一時スクリプトに60秒の上限を設定。システム Python には PyYAML がなかったため既存環境へ切り替えた。依存追加はしていない。
- 今回は文書変更のみ。戦略テスト・バックテスト・実注文は未実行。
- 独立した読み合わせで4つの依頼例（README修正、関数分割、新パラメータ比較、注文後timeout診断）を検証。Skill 選択・作業範囲・完了条件が意図に合うことを確認し、挙動維持テスト、診断時の停止権限、DSR計算不能時の採否の3点をさらに明確化した。これは行動方針の読解検証で、4タスクの実行試験ではない。

## 範囲外に残る確認事項

以下はコード修正を行ったという意味ではない。Skill が誤った現状説明をしないために、実装上の限界として記録する。

- `src/research/experiment_utils.py` の集計定義（フラット日除外、DD定義、同名試行数）と、DSR 計算の Sharpe 単位・試行範囲は性能評価前に別途検証が必要。今回 helper 自体は変更していない。
- `src/leadlag/models/v2/audit_comparator.py::_run_safety_audits` は数値 FAILED 時にフラット化するが、リーク FAILED を同じ箇所でフラット化しない。本番の全下流経路で確実に発注停止するかは、この文書レビューでは保証していない。
- `docs/refactor_roadmap.md` は未チェックボックスがなくても、部分完了の表・当時の設計・後段の完了注記が混在する。ADR-P35 と関連 ADR を確認したが、全 Phase の再監査・更新は本依頼の範囲外。今回の完了は指示文書の見直しに限定する。
