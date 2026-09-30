# 09:10入力・実口座損失ガード・ML forward評価の実装記録

**実施日:** 2026-09-29  
**対象:** 改善分析 A2、D1、E1/E2、および実装優先表の2〜4  
**設計判断:** [2026-09-29 decision](../../docs/decisions/2026-09-29-frozen-0910-account-risk-forward-eval.md)

## 実装内容

### 2. 09:10 quoteの記録と凍結

- 立花のmarket-price応答をローカル受信時刻付きで取得し、JP17銘柄とTOPIXの両側気配、trade date、9:10:00〜9:10:30の窓を検証する不変snapshotを追加。
- 最初の完全な有効snapshotを日付ファイルへ一度だけ保存し、snapshot IDと銘柄別観測時刻をdecision input fingerprintに含める。
- gap生成とactual-live decisionを同じfrozen snapshotへ接続し、実行manifestとML shadowへIDを残す。当日データが欠ける場合の価格cache・前日終値への代替は許さない。
- capture、gap生成、decisionで `LEADLAG_CAPTURE_OUTPUT_DIR` を共有する。

### 3. 実口座損失stopと照合

- prior sessionまでの実口座daily/monthly returnを検証する `account-risk-snapshot-v1` 契約を追加。trade date・account key・観測時刻、PnL basis、cash・position・fee照合の完了を検証してから既存の日次/月次閾値へ渡す。
- actual-live時にsnapshotが無効・欠落している場合も新規リスクを停止し、既存のpost-decision判定で確認済み在庫を減らす注文だけを許す。
- 実口座PnL snapshotを生成する検証済みproducerは未実装。ユーザー確認により現時点で正本データ源はなく、fail-closed継続を決定した。既存の立花受入保証金を損益として流用せず、適格なsnapshotが用意されるまでactual-liveの新規建てはfail-closedとなる。producerは正本と照合責任が定まってから着手する。

### 4. ML forward評価・レジストリ訂正

- 実験レジストリに安定ID、`study_id`、metric schema version、訂正元・置換先を追加。訂正は追記のみで、現行viewではsupersedeされた旧記録を除外する。試行数は系列IDを指定して数え、訂正行は試行に含めない。
- 既存registryに残っていた長期感応度10候補の旧`rank_ic`を訂正。旧値は `rank_ic_open_to_0910_diagnostic`、正しい予測target値は `rank_ic_vs_backtest_target` と別名にして、元のJSONL行を保ったまま訂正recordを追加した。値は `var/results/20260924_sensitivity_pipeline_audit_long/rank_ic_vs_backtest_target.csv` と対応し、metric schema `sensitivity-rank-ic-target-v2` / study `sensitivity-pipeline-audit-long-2026-09-24` で記録した。補正targetは09:10 quote proxy 3.13%と始値→大引けfallbackを混ぜた回顧系列であり、実測quote主体の前向き検証ではない。
- ML有効/無効shadowと日次outcomeをtrade date・input digest・quote snapshot IDで照合する評価器とCLIを追加。frozen 09:10 midpointから公式終値までのラベルだけを受け入れ、missing、quote不一致、複数入力版、ラベル不足を明示して保持する。
- モデル費用を使ったnet proxyと、約定・建玉・cash・feeの完全照合後だけ算出するML有効側の実PnLを別表示する。
- 現在のproduction `CURRENT` は、要求cutoff `2025-12-31`、実際の `train_end` / `label_asof_end` `2025-12-30` のartifact（12/31は東証休場）を指す。したがって2026年の取引日は日付上は学習後で、固定済みartifactから当時生成・保存した予測があればOOS候補になる。
- ただし2026-09-27にこのartifactで2025-12-31〜2026-09-25の170営業日を再計算したVaR履歴は回顧再生であり、独立した前向きOOS評価には数えない。promotion記録ではcutoff再構築後の前向きlabelは0日。`daily.jsonl` / `outcomes.jsonl` に有効なpaired forward記録はなく、09/28のquote captureもAPI v4r9の404で失敗している。よって「2026年データ自体が学習期間外に存在しない」のではなく、「新artifactを事前固定した後の有効なpaired forward観測がまだない」が正確な状態である。
- 既存の [MLコスト感度分析](../20260927_ml_overlay_cost_sensitivity/report.md) には2024-12-23〜2026-07-29の368日分のML有効/無効pairedモデル損益があり、5/10/20bpsのコスト感度とpaired bootstrapを計算済み。ただし、これは学習終端2024-12-20の旧artifactで作成済みのStage 5/6出力を再価格付けした既知期間分析で、実約定費用を使わず、今回の2025年末cutoff artifactの評価にも、事前登録した250日forward gateにも代用できない。
- 後続分析として、同じ269日・同じ入力からML有効／無効を比較するversion-awareな回顧paired replayを実施した（旧artifact 99日、2025年末cutoff版170日）。全体では複利net差+3.20pp、Sharpe差−0.057、最大DDは1.46pp悪化し、paired平均net差の95% block-bootstrap区間は0を含んだ。cutoff版の170日区間単独では複利net差−0.41ppだった。詳細・全日データは[269日paired replay report](../20260929_ml_overlay_paired_269/report.md)に記録した。
- これは保存済みの前回ML有効VaR履歴を日次returnで完全再現したが、今回のgap snapshot fingerprintは前回記録値と異なった。provider `available_at` とfrozen 09:10 quoteの裏付けも不足しているため、独立OOS・実費評価・現行artifact固定の事前登録250日forward gateには数えない。旧版99日を足して現行artifact固定の250日基準を満たしたことにもならない。
- baselineの反実仮想execution cost replay、現行artifact固定の事前登録250日forward paired gate、そのbootstrap・採否判定は未実施であり、評価ツールの実装だけをoverlayの有効性証明とは扱わない。
- 公式v4.10の `CLMMfdsGetMarketPriceHistory` を読み取り、JP17の指定日 `pDPP` 終値をfrozen 09:10 midpointと結合するCLIを追加した。公式仕様の履歴更新時間に合わせ翌日01:00 JSTより前の実行を拒否し、全銘柄・同一日付の終値が揃わないと保存しない。outcomeは日付・input digestごとに追記専用で、変更値への上書きを拒否する。認証・API通信・scheduler登録は行っていない。
- このcollectorが出すのは価格targetのみで、約定・cash・feeを埋めない。ユーザー確認により実口座PnLの正本は現時点で存在せず、actual-liveの新規エクスポージャーは引き続きfail-closedである。
- 複数日評価時に全日付へ最後に読み込んだoutcomeのsource名を付ける不具合も修正した。異なるsource名の2日分の正常ペアを使い、各日のinputに一致するsourceが保存されることを回帰確認した。

### 09/28 capture障害の原因と修正

- 09/28の既存capture・decisionログは、立花API v4r9の認証URLに対するHTTP 404で停止していた。立花証券の公式告知はv4r9を2026-09-27廃止、v4r10を現行版としている。([告知](https://www.e-shiten.jp/e_api/)、[v4r10仕様](https://www.e-shiten.jp/e_api/mfds_json_api_ref_text.html))
- v4r10の公式仕様で `CLMMfdsGetMarketPrice` と既存の口座・信用建玉・注文照会APIを確認し、v4r10で廃止された旧マスタ一括取得・ニュースAPIは現行ブローカーアダプタから呼ばれていないことを照合した。
- 本番・デモの既定URL、環境設定例、接続診断スクリプトと回帰テストをv4r10へ更新した。API認証・口座照会・本番通信はこの作業では実行していないため、エンドツーエンド復旧は未確認である。
- v4.10の履歴仕様は日次終値 `pDPP` と当日キー `sDate` を返し、前営業日の情報は00:00〜00:59 JSTに更新され、01:00以降の取得が指定されている。この契約だけを利用する読み取り専用outcome collectorを実装した。公式仕様に従う有効時刻・全17銘柄の一意行・固定quote IDをオフラインfixtureで検証した。
- `.env.example` にあった認証ID・第二パスワード欄の値を除去し、空欄のテンプレートにした。追跡中のHEADにも値が残っていたため、実資格情報なら失効・再発行が必要である。過去Git履歴はこの作業では書き換えていない。

## 検証結果

- 対象回帰テスト: **129 passed**（quote snapshot、capture、input contract、account risk、ML forward/shadow/outcome collection、registry/utilities、runtime、order lifecycle、Tachibana API version/config）。
- リポジトリ全体: **954 passed**（最終ツリー、`tests/`、xdist 10 workers）。既存のPerformanceWarning（DataFrame fragmentation）とConstantInputWarning（定数配列の相関計算）が2件。
- `compileall`: 成功。
- Ruff: 成功。
- mypy: 156 source filesで問題なし。
- import-linter: 195 files / 562 dependencies、7 contract維持、違反0。

quote収集の実市場実行、実口座snapshotの生成、ライブ発注、OS schedulerへのplist登録は行っていない。schedulerに入れる前に、account-risk producerの実データ契約・照合責任者と、quote captureが決定時刻に全18銘柄を安定して取得できることを運用環境で検証する必要がある。
