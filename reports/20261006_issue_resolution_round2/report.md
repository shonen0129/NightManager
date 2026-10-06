# 監査issue追加対応（2026-10-06）

前回の修正commit `fd82e99e` を起点に、#35/#44/#46のローカル実装を進めた。開始時worktreeはcleanだった。元の[監査報告](../20261006_project_audit/report.md)と[前回の対応報告](../20261006_issue_resolution/report.md)は保持する。GitHubへの投稿・close、実発注、broker照会、実Yahoo更新、live cache書換え、scheduler起動は行っていない。

## 今回の到達点

| Issue | 実装・検証した範囲 | 残件 |
|---|---|---|
| #35 | ADR計算/取得を `data.adr_producer` へ移動し研究入口を撤去。US/JP日付対応、補間しないreturn、構造的ゼロと取得欠損、coverageを検証。pickle/CSV/manifestをZIPとして原子的公開し、hash・日付・coverage不整合を拒否。scheduled updaterは当日行を要求し、失敗時exit 1、直前bundle保持、次の正常更新で復旧。単独復旧CLI、CIのbatch Python import境界、運用手順を追加 | 実ソースでの当日更新復旧・日次監視受入。shell起動のdistribution/vol diagnosticsはまだresearch model/V1 backtest依存であり、運用全体の分離完了とはしない |
| #44 | BLPXの7計算（inverse、ridge係数、PCA prior、Tikhonov、confidence、非対称solve、診断構築）を明示入力の `core.blpx_math` へ統合。両modelの重複methodと旧再公開を撤去。研究診断キーを `z_U_t` へ揃えた。疑似逆行列の非有限結果を検出する回帰修正も実施 | rolling/window・sector prior・signal/predict orchestration等の重複は残る。監査の14 clone group全件を解消したとはしない |
| #46 | `BacktestEngine._simulate_daily_pnl` を撤去し本番・研究callerを `core.pnl.simulate_daily_pnl` へ更新。3値 `load_gap_matrices` を撤去しtestを4値bundle契約へ更新、metadata gateを維持。testしか使用元がないCostCalculatorを撤去し、実用bps費用関数の回帰へ置換 | 未接続provider、研究専用convex optimizerの配置、研究用ML wrapperが残る。前回撤去したreports watchdog依存も引き続き解消済み |

いずれもissue全体をcloseする判定ではない。#33の終端在庫・数量/cash会計・連続replay、#37の損益/リスク再評価、#41のstudy履歴、#45の大きいorchestration分割、#34/#47の未確認部分、#27/#25/#23/#18の外部・時間依存の受入は今回完了としていない。

## ADRの品質・公開契約

正規保存先はdeployment runtime root配下の `data/adr_features.zip`。旧pickleへのfallbackと自動変換を置かず、実closeからの再生成を必要とする。旧artifactのゼロ補間結果を新coverageとして認定しない。そのため、正常な新bundle公開前はMLがskipする。

bundleには `features.pkl`、`features.csv`、schema version 1の `manifest.json` を保存する。manifestは公開時刻、最終trade/signal日、historical不完全行数、当日sector coverage、payload hashを持つ。対象日が最終trade日と一致し、全17特徴が有限で、対応ADR観測がsectorごとに1以上あり、signal日がtrade日より前の場合だけcommitする。未対応sectorはzero/coverage=0を要求する。部分取得時は観測されたADRだけで平均し、観測数を隠さない。

同一directoryのtemporary fileをfsyncしてos.replaceするため、CSVだけ新しくなる世代混在を公開しない。公開前の例外・破損・必要日の欠損で正常扱いにしない。historical NaNを残し、その日のoverlayやADR付き学習はskipする。daemon wait timeout後の遅延download threadが公開する経路はない。network timeoutだけをwhole-process deadlineとは扱わず、既存job guard/phase deadlineを維持する。

日付の前後関係と保存coverageは、Yahooが9:10以前に提供したという時点可用性の証明ではない。また、全ADRの取得成功や実運用SLOの達成を今回の合成fixtureから認定しない。

部分欠損時の平均は旧zero-imputed平均から変わる。既存ML artifactの再学習・OOS性能比較は今回実施しておらず、新しい入力の分布と既存artifactの整合は本番受入の残件とする。wheelの推論smoke成功は収益性能の認定ではない。

## 独立版比較と回帰

[比較スクリプト](../../src/research/scripts/experiments/verify_audit_round2_refactor.py)はgitから変更前コードを取得し、別module・別model instance・独立config/input copyで比較する。[比較結果](comparison.json)は次を記録する。

- 本番/研究BLPX計16ケース（scalar/covariance、prior scaling有/無、通常/非有限入力窓）の数値出力を誤差0で比較。未知の当日JP target・未来JP/US行の摂動でも当日出力が不変。
- PnL 3ケース（持越率0/0、0.75/0.5、1/1）で全日次return/cost/turnover/exposure出力が誤差0で一致。初期/終端会計を新たに検証済みと主張しない。
- 特異/finite/NaN/Inf逆行列4ケースを比較。旧2×2 Inf入力でのNaN結果1ケースはゼロ行列・fallback=trueへ修正したため、意図した挙動差として記録。
- 観測が揃ったADR入力では、旧生成器の全特徴とsignal日が一致。欠損のゼロ化撤去は挙動変更として回帰で別に検証。

ADRの回帰はmissing sector/signal day/invalid price、duplicate/NaT/逆順/当日signal、未来価格・未知target摂動、歴史欠損、payload/manifest破損、false coverage、公開例外後の保持・復旧、scheduled Python入口を通す成功/失敗を含む。旧APIの自己比較testは撤去し、実数学・原子的公開・費用の契約を直接検証する。

## 検証

最終結果は [verification.json](verification.json) に保存する。回帰baselineを単独実行後、残りの `tests/` 全体を4 workerで検証する。すべての長時間commandへ外側のtimeoutと終了猶予を設定した。

最終全体検証はbaseline 1件と残り1,066件、合計1,067件が成功（既存warning 3件）。対象回帰107件、追加境界64件、費用13件、index名修正後のADR関連45件も成功した。最終whole-suiteはindex名修正とcaller import整合後に再実行した。

```bash
timeout -k 10s 120s .venv/bin/python src/research/scripts/experiments/verify_audit_round2_refactor.py \
  --baseline fd82e99e --output reports/20261006_issue_resolution_round2/comparison.json
timeout -k 10s 300s .venv/bin/python -m pytest tests/regression/test_v2_baseline.py
timeout -k 30s 1800s .venv/bin/python -m pytest tests --ignore=tests/regression/test_v2_baseline.py -n 4
```

compileallは `src/leadlag tests tools scripts src/research` の全5 tree、RuffはCI対象と追加producer/比較入口、mypyはproduction package、import-linterは7契約、scheduled Python入口は2ファイルを検証する。変更した研究17ファイルのRuffも変更前後で比較し、既存11件を2件へ減らし、新規diagnosticは0件。[詳細](research_lint_comparison.json)。残るF841/F541は元からある `experiment_order_cost_gate_20260923.py` のもの。

wheelは既存cacheのsetuptools/wheelを読み取り、一時project copyからoffline buildする。production source manifestとの一致・research除外、隔離wheel import、temporary deployment root、ADR bundle生成/公開/読込、共有BLPX math、CLI help、ML artifact inferenceを確認する。環境への依存追加は行わない。setuptoolsの既存license TOML形式のdeprecation warningは残る。

設計判断は [ADR](../../docs/decisions/2026-10-06-adr-publication-and-shared-blpx.md)、ADR単独復旧のlease/deadline付きcommandは [運用手順](../../docs/日次運用手順書.md) を参照。新artifactが実運用で生成されたとの報告はしていない。
