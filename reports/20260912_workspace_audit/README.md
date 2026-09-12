2026-09-12 workspace audit の再現資料

本体は [report.md](/Users/shonen/leadlag/reports/20260912_workspace_audit/report.md) です。本番ソース・設定の修正を行う前の監査結果です。以下のスクリプトは調査用で、標準回帰テストへの追加は行っていません。

| 入力・スクリプト | 保存結果 | 対象 |
|---|---|---|
| `inspect_workspace.py` | `inventory.json` | 解決済み設定、構造、storeの概況 |
| `inspect_data.py` | `data_summary.json` | ローカルデータのread-only集計 |
| `reproduce_findings.py` | `reproductions.json` | 監査失敗、PnL、MDD、DSR、cache、atomicity、PIT不足 |
| `reproduce_integration.py` | `integration_reproductions.json` | 偽broker、risk停止、古い日付、休日、ML省略、VaR設定 |
| `probe_model.py` | `model_probes.json` | 合成履歴でhorizon cacheと未来入力摂動 |
| `probe_broker_contract.py` | `broker_contract_probes.json` | 立花API応答fixture、FILLEDの約定価格収集 |
| `probe_data_contract.py` | `data_contract_probes.json` | 正常入力のstrictエラー、価格の意味、Step2設定 |
| `summarize_checks.py` | `check_summary.json` | inventory・Ruff・ASTの集計 |
| `validate_artifacts.py` | `artifact_validation.json`, `source_manifest.json` | JSON、ソース参照、検証対象hash |

これらは本番の注文を送信しません。broker関連の再現では偽brokerかモックを使用し、temporary storeを使用します。`inspect_data.py` は手元のstoreから集計するため、後日データが変われば結果も変わります。モデルprobeはmacro・ML等を無効化した隔離設定を使用します。本番設定ファイルは変更しません。

再現には既存の `.venv` を使用します。リポジトリルートから実行してください。以下は出力ファイルを上書きせず、再現結果を標準出力へ出します。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/reproduce_findings.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/reproduce_integration.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_model.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_broker_contract.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_data_contract.py
```

標準テスト・静的検証は次の範囲を実施しました。全体テストの結果は576 passed / 1 failedです。標準テストにはローカルcacheに依存するregressionがあるため、同じcommitでも環境で結果が変わる問題をF22に記録しています。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n 4 --tb=short -q
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python -m pytest tests/regression/test_v2_baseline.py --tb=short -q
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python -m compileall src/leadlag tests tools scripts src/research
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/ruff check src/leadlag src/research tests tools scripts --output-format json
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/mypy src/leadlag
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/lint-imports
```

標準テスト内のネットワーク/cache依存がすべて除去済みとは保証していません。今回の最小再現は依存箇所をモックまたは合成入力で置換しています。数値probeのJSON出力は観測事実であり、バグ修正後は値が変わります。修正時には期待挙動をassertする標準回帰テストへ移す必要があります。

`watchdog.py` は指定秒数で子プロセス群をTERM、終了猶予10秒後にKILLします。ツールの `yield_time_ms` を停止期限の代わりにしていません。ログ中の `WATCHDOG: exit=...` はこの外側の終了状態です。

公式外部資料は2026-09-12に確認しています。立花APIの注文応答は [公式仕様](https://www.e-shiten.jp/e_api/mfds_json_api_ref_text.html)、取消後の照合は [公式FAQ](https://mobile.e-shiten.jp/QA/answer14.html)、DSRは [原論文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)、市場時刻は [JPX取引時間](https://www.jpx.co.jp/english/equities/trading/domestic/01.html) と [売買制度](https://www.jpx.co.jp/english/equities/trading/domestic/04.html) を参照しました。

`source_manifest.json` はレポートで参照した既存ファイルのhashを保存します。監査開始前からworkspaceにはユーザーの未commit変更があるため、git revisionだけでは内容を一意に再現できません。秘密情報を含む設定・口座情報の全ファイルコピーは作成していません。
