# 09:10入力の凍結、実口座損失ガード、ML forward照合（2026-09-29）

## 決定

actual-liveの判断・gap生成が使う当日価格を、立花のJP17銘柄とTOPIXの両側気配から作る日次不変snapshotへ統一する。取得対象は `CLMMfdsGetMarketPrice`、利用可能時刻はローカル受信時刻、取得窓は09:10:00〜09:10:30 JSTとする。完全な18銘柄cross-section以外は確定せず、同じtrade dateの初回有効snapshotを `O_EXCL` で保存する。snapshot IDと銘柄別受信時刻を `KnownMarketInputs` のfingerprintへ含める。

既定保存先は `var/shadow_runs/ml_overlay_value/microstructure/frozen_YYYYMMDD.json`。`LEADLAG_CAPTURE_OUTPUT_DIR` を使う場合、capture・gap生成・decisionに同じ値を渡す。当日ファイルが存在しない、壊れている、観測時刻が条件外、または必要銘柄のquoteが欠ける場合、actual-liveは中止し、古い価格cache・前日終値を使って執行しない。従来のshadow-only capture失敗許容はshadow保存エラーに限り、必須live入力の失敗には適用しない。

モデルreplay損益とは別に実口座の損失stopを設ける。actual-liveはprior-sessionまでの口座損益snapshotが `account-risk-snapshot-v1` 契約を満たす場合だけ既存の日次・月次損失閾値へ渡す。口座IDと対象日を照合し、cash・建玉・費用の照合完了、既知費用を反映したPnL基準、decision cutoff以前の観測時刻を必須とする。停止超過時またはsnapshot不在・無効時は新規リスク増加を止める。既存ポジションの縮小・決済は既存post-decision方針に従う。

append-onlyの実験レジストリでは、記録修正を削除・上書きせず `record_id`、`correction_of`、`supersedes`、`metric_schema_version`、`study_id` で関連付ける。現行viewは訂正元を除外し、試行回数は同じ `study_id` の実験試行だけを数える。過去試行の未登録を数え直す場合は `study_id` と対象期間を明示し、暗黙の自動推定で過小評価しない。

実装時に、2026-09-24長期感応度auditの10候補についてregistryの旧 `rank_ic` を確認した。各旧metricはopen→09:10診断値だったため、旧値を `rank_ic_open_to_0910_diagnostic` として区別し、付属CSVのtarget一致Rank ICを `rank_ic_vs_backtest_target` として訂正recordへ追記した。元JSONL行は維持し、study IDを補って試行数を10件として数えられるようにした。訂正targetもquote proxy 3.13%と始値→大引けfallbackを混ぜた回顧系列で、forward実測ではない。

ML有効/無効のforward評価は日付だけでなく入力digestとfrozen quote IDを照合する。targetは凍結09:10 midpointから公式終値までの系列とし、日付抜け・quote不一致・同日複数input versionを明示したまま保存する。ML有効側の実PnLとモデル費用によるnet proxyを区別する。candidateの実PnLはfill・inventory・cash・feesが照合済みで、同一input digestを確認できた場合にだけ表示する。baselineの反実仮想執行費用が揃わない限り、実費ベースの採否判断は行わない。

price targetの前向き収集には、v4.10 `CLMMfdsGetMarketPriceHistory` の日付付き終値 `pDPP` を使う。公式仕様上、前営業日のhistoryは00:00〜00:59 JSTに更新され、01:00以降に取得する。frozen quote snapshotのJP17 midpointと当日終値を使ってtarget returnを作り、日付・input digest・quote ID・各終値row hashを追記専用`outcomes.jsonl`へ保存する。欠損や複数行、別日のrowは一件でもあれば作成を拒否する。実約定・口座損益は価格ラベルと別データである。

2026-09-28のcapture/decisionログで、立花API v4r9認証先のHTTP 404を確認した。公式告知上、v4r9は2026-09-27に廃止済みであり、v4r10仕様を現行版としているため、コード既定値・デモ設定例・診断ツールをv4r10に更新した。現行アダプタで呼ぶmarket-price、口座・建玉、注文照会API識別子はv4r10の仕様に残り、v4r10で廃止されたマスタ一括取得や旧ニュース取得IFは使用していない。認証・実口座照会を含む復旧確認はしていない。`.env.example` の認証ID・第二パスワード値も除去し、利用者がローカル設定する空欄にした。追跡中のHEADにも値が残っていたため、実資格情報なら失効・再発行が必要である。Git履歴は書き換えていない。

## 理由と安全境界

9:10の価格cacheだけでは、異なる時刻の価格が同じ入力として扱われる。再取得・後日の訂正で当日の判断を変えず、gap計算・判断・shadowの入力対応を追跡するため、availabilityの根拠とsnapshot IDを一緒に固定する。

実口座の受入保証金やモデルreplay returnは、現金・建玉・全費用の照合済み実損益を意味しない。確認できない値を損失stopの真値に使うと誤検知・見逃しの双方が起きるため、適格な口座台帳がない場合はfail-closedにする。

## 実装状態と未完了

- quote snapshot検証・不変保存、capture・gap・actual-live decisionへの接続を実装した。
- account-risk snapshotの検証と日次/月次停止判定を実装し、actual-liveで証跡が欠落・不適格なら新規エクスポージャーを止める。
- **照合済みaccount-risk snapshotを生成するproducerは未実装。** ユーザー確認により現時点で正本データ源はなく、fail-closed継続を決定した。適格なsnapshotが供給されるまでactual-liveの新規建ては停止状態となる。既存の受入保証金をPnLへ転用しない。
- ML shadow結果・quote ID・実行planの入力digestを保存し、forward outcome照合器と翌日01:00 JST以降に公式終値を集める手動CLIを追加した。現在のproduction `CURRENT` が指すartifactは要求cutoff `2025-12-31`、実際の `train_end` / `label_asof_end` `2025-12-30`（12/31は東証休場）。したがって2026年は日付上OOS候補だが、2026-09-27に後から再計算した170営業日の履歴は回顧再生であり、独立した前向きOOSには数えない。cutoff再構築後の前向きlabelは0日で、有効なpaired forward記録もまだない。過去には旧artifact（train end 2024-12-20）の368日pairedモデル損益・コスト感度分析があるが、実約定費用を使わず、現行cutoff版または今回の事前登録250日forward gateとは別評価である。cutoff版VaR履歴269日は旧版99日＋cutoff版170日のoverlay有効側だけなので、現行artifact固定のpaired標本にはできない。baseline反実仮想約定の完全なcost replay、現行artifact固定の事前登録250日forward paired gateは未完了であり、今回の変更はML overlayの有効性を採用判定しない。
- 後続の研究診断では、2025-07-29〜2026-09-25の269営業日を、production `HISTORY.json` が割り当てる旧artifact 99日／cutoff版170日に分け、同一入力のML有効／無効で回顧paired replayした。全体とcutoff版区間の結果・fingerprint差・限界は[269日paired replay report](../../reports/20260929_ml_overlay_paired_269/report.md)に記録した。前向き250日gateや採否判定とは区別する。
- 指定trade dateの終値targetをbroker read-only APIから集める手動CLIを追加した。翌日01:00 JSTより前は実行を拒否し、全JP17銘柄が揃うまで追記しない。実API接続は試しておらず、scheduler登録もしていない。
- ユーザー確認により実口座PnLの正本ソースはまだないため、照合済みaccount-risk producerは保留し、actual-liveの新規リスクはfail-closedを継続する。
- 変更したplistをOSのschedulerへ登録していない。ライブ発注・注文キャンセル・再送は実行していない。

## 検証

APIバージョン修正、公式終値collector、複数日評価器修正を含む最終検証（対象129件、全954件、compileall、Ruff、mypy 156 files、import-linter 7 contracts）は [実装記録](../../reports/20260929_frozen_0910_account_risk_forward_eval/report.md) に記録した。口座risk producer、paired forward実現値、実約定との照合を要する検証は別の運用データが必要なため未実施とする。
