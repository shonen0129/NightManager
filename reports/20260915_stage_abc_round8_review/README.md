**段階A〜C 第8回レビュー（2026-09-15）**

[詳細レビュー](./review.md)に判定、X01/X02の受入結果、F01〜F22・後続レビューの再発確認、運用未了と検証限界を記載しています。[機械可読サマリー](./review_summary.json)と[証拠検証](./validation.json)も参照できます。

レビューのみを行い、本番ソース・設定・既存テストを変更していません。前回の証拠は修正時に更新されていたため、新しいディレクトリで再実行しています。[開始snapshot](./source_snapshot.json)と[終了manifest](./review_manifest.json)で対象hashを追跡できます。

再現はリポジトリルートで、既存環境とprocess groupの停止期限を使用します。

```bash
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260915_stage_abc_round8_review/probe_provenance.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260915_stage_abc_round8_review/probe_boundaries.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260915_stage_abc_round8_review/probe_deadline.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260915_stage_abc_round8_review/probe_additional.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n auto --tb=short -q
```

過去指摘の再現は`probe_review.py`、`probe_schema_cache.py`、`probe_var_gap_version.py`、`probe_contracts.py`にあります。`model_probes.log`は`reports/20260912_workspace_audit/probe_model.py`、`prior_probes.log`は`reports/20260913_stage_abc_rereview/probe_remaining.py`を現行コードで実行した出力です。元probeへの依存があるため、このディレクトリだけを取り出して実行する用途ではありません。

`probe_additional.py`はコピー・config読込・cache初期化の待機、実SQLite書込ロック、保存子プロセスの正常終了・異常終了・回収を追加確認します。故障注入は一時データに限定します。ログ内のシリアライズTypeErrorは意図した障害で、期待する親側例外と既存値保持を検査しています。

`prepare_workspace.py`はソースhash取得、`prepare_report.py`は完了済み全テストからのサマリー作成、`validate_review.py`は値・契約・文書リンク・hashの整合確認です。再実行するとこのディレクトリ内の証拠を更新するため、次回レビューでは別ディレクトリを使用してください。
