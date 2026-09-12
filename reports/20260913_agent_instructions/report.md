# AGENTS.md / Skill の作業範囲・参照条件の見直し

作成日: 2026-09-13

## 対象と方針

ユーザー指定の [OpenAI 記事](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) を参照し、AGENTS.md と `.agents/skills/` の10個を修正した。適用条件の明確化、必要時だけの詳細参照、依頼範囲内での完了を重視した。設計判断は [ADR](../../docs/decisions/2026-09-13-scoped-agent-instructions.md) に記録した。

対象ファイルには作業開始前から未コミット変更があり、`debugging` は未追跡だった。その作業ツリーを保存して比較基準とし、反映前に内容が変わっていないことを確認した。既存のコード・設定・レポートと `.devin/skills/debugging/SKILL.md` の削除状態は今回の変更に含めていない。

## 主な変更

| 対象 | 変更 |
|---|---|
| AGENTS.md | 設計文書を用途別に参照。ロードマップ照合はリファクタリング・Phase 進捗時に限定。依頼範囲の修正・検証を継続する条件と `debugging` の案内を追加 |
| 10個の description | 実際に行う作業を短く記述。広いキーワードや「必ず参照」による過剰適用を整理 |
| debugging | 共通規約の重複と固定の6ステップ・全件チェックを整理。フラット/PIT、データ/キャッシュ、指標/トレースを3つの参照文書へ分離 |
| code-review / refactor | 本番設定・設計文書・別 Skill のレビューを、対象の評価に必要な場合に参照 |
| leadlag-fund-improvement / leak-audit / test-gen | 対象設定・計算経路・未検証の契約に応じて観点を選ぶ条件を明示 |
| hang-prevention | 停止期限を付けるだけの作業と既存障害の復旧を分け、スケジューラ確認を日次実行時に限定 |

単純な文字数削減を優先せず、データ整合規約・8つの不変条件・改善ワークフロー・評価指標と注意点は全文を保持した。実行コードや挙動に関わる設定変更後の全テスト、注文権限、監査失敗時の扱い、感度分析・DSR・OOS・shadow の要件も維持している。

## debugging の現行実装との照合

- 存在しない `prod-backtest-consistency` への誘導を、既存の V2 経路・shadow 確認へ置き換えた。旧 `scripts/experiments/` の実行例は外し、正本 `BacktestEngine.run_v2_backtest()` を案内する。
- 日次バッチには実発注が含まれるため、原因調査でそのまま再実行させる手順を削除。保存入力・fixture と副作用の確認を使う。無条件のプロセス停止・lock 削除・再送は `hang-prevention` の現行規約へ統一した。
- `preprocessor.py::preprocess_data` の全列/一部欠損、有効数、strict、暫定行の分岐を確認。「一銘柄の NaN で必ず全行スキップ」という古い説明を修正した。
- `decision_cache.py::is_decision_cache_valid` は双方の `updated_at` があれば比較し、不足時に mtime を使う。mtime のみを正本とする説明を修正した。
- cache → on-demand → flat と PIT multiplier を分離。履歴数・MinVar 係数・overlay・ログパス・稼働時刻は有効設定と実際の経路へ照合するよう変更した。
- シグナルトレースでは、当日既知の US リターン・gap と、未知の当日ターゲット・未来行を区別した。旧固定の fallback 5%、sign agreement、RMSE を普遍的な合格基準にしない。

## 検証

- `skill-creator/scripts/quick_validate.py` の `validate_skill()` で10個すべての YAML frontmatter・命名・未完成プレースホルダー検査: PASS。
- AGENTS.md、10個の Skill、3個の参照文書について、相対リンク9件と具体的なリポジトリパス49件の存在、末尾改行・空白・コードフェンス: PASS。生成先パスの将来の実在性や全シンボルの意味を保証する検査ではない。
- 保存した開始時点との比較で、データ整合規約、不変条件8項目、改善ワークフロー、評価指標と注意点の全文一致: PASS。
- 今回の差分の空白検査と反映内容の照合: PASS。
- ADR-P33・P34・P35 と `docs/refactor_roadmap.md` の関連箇所・未了表記を照合。Phase 全体の実装監査・完了判定は対象外。

`skill-creator` の Independent Forward-Testing 手順に基づき、独立エージェントが次の5依頼を文書だけで読み合わせた。ファイル変更・プロジェクトCLI・本番アクセスを伴う実行試験ではない。

| 依頼例 | 読解結果 |
|---|---|
| README の誤字1箇所 | 文書の差分・文意確認で完了。追加の戦略調査・全テストは要求されない |
| 当日 cache 不在、on-demand 依存なし、PIT multiplier 0.5 | 終端フラットの直接原因と PIT 履歴不足を分離し、調査のみの範囲を維持 |
| 出力を保つ weight 関数分割 | 独立した変更前後比較と全体回帰を実施する条件。性能実験へ拡張しない |
| 注文送信後のタイムアウト調査 | 注文状態・残存処理を調査。停止・再送を自動的に実行しない |
| OOS 未実行の既存結果を Markdown 化 | 未実施と明記して報告。追加 OOS やスイープを必須にしない |

この5例で、矛盾による停止や依頼外の操作を必須とする指示は見つからなかった。モデルによる実タスクの成功率・速度・消費トークンの比較は未実施。

## 文書量と検証範囲

| 指標 | 開始時 | 改訂後 |
|---|---:|---:|
| 10個の description の合計文字数 | 857 | 418 |
| 10個の SKILL.md の合計行数 | 414 | 327 |
| debugging の SKILL.md 行数 | 126 | 35 |
| AGENTS.md 行数 | 94 | 96 |

文字数は Unicode 文字数でありトークン数ではない。Skill の合計行数には新しい参照文書を含めない。文書・Skill のみを変更したため、戦略テスト・バックテスト・本番実行は行っていない。
