# 段階A〜C 第5回・再発確認レビュー — 2026-09-14

**判定：BLOCK。P2が4件残っています。** 前回U02〜U04の修正は再現確認できました。U01も通常の`decide`経路は改善していますが、互換APIを使う経路に検証結果の伝播漏れがあります。元の指摘まで戻ると、注文状態変換と入力cacheにも未解消の境界がありました。

今回の4件は「前回の修正が全部元に戻った」という意味ではありません。**修正済みの主要ケースの再発は確認せず、別の入口・入力条件に残る問題を再現しました。** 本番artifactの再生成・本番/BTの同一weights照合という既知の運用未了は、4件とは別に扱います。

対象はHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285`上の未commit作業ツリーです。元の[F01〜F22](../20260912_workspace_audit/report.md)、[R01〜R08](../20260913_stage_abc_review/review.md)、[S01〜S05](../20260913_stage_abc_rereview/review.md)、[T01〜T05](../20260913_stage_abc_round3_review/review.md)、[U01〜U04](../20260914_stage_abc_round4_review/review.md)、[今回の修正報告](../20260914_stage_abc_round4_fix/fix_report.md)を照合しました。追跡対象の変更は54ファイル、前回manifestからの更新・追加は14ファイルです。

全体テストは **620 passed / 17 warnings / 378.67秒**。Ruff・mypy・compileall・import-linterも成功しました。本番ソース・設定・テストをレビューのために変更していません。検証は偽broker・一時DB・一時artifact・固定fixtureで行い、実口座への注文・決済・照会は行っていません。

**V01 / P2 — 不整合な`.npy` bundleが、3値の互換loaderを経由すると引き続き使用されます（F12／T03／U01）。**

対象：[gap_matrix_io.py:405](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:405)、[同:421](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:421)、[同:436](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:436)。利用側：[gap_io.py:292](/Users/shonen/leadlag/src/leadlag/models/v2/gap_io.py:292)、[signal_enhancement.py:89](/Users/shonen/leadlag/src/leadlag/models/signal_enhancement.py:89)。

manifestのdigest照合自体は機能しています。ただし非strict時は不整合を検出しても配列を返し、metadataだけをNoneにします。3値の`load_gap_matrices`はmetadataを捨てるので、不整合の判断がalert文字列にしか残りません。`compute_distribution`の`_gap_alerts_fatal`は`[FATAL]`だけを検査するため、新しいconsistency alertを無視します。`apply_multi_horizon_blend`はalert自体を捨てます。

**実害のある到達先：** `compute_distribution`→研究の`_multi_horizon_scores`利用箇所に加え、[Step 2のbaseline IR:1015](/Users/shonen/leadlag/tools/research/compute_gap_adjusted_distribution.py:1015)が`apply_multi_horizon_blend`を使います。正常な本番`decide`で今回も混在分布がPASSEDになった、という指摘ではありません。同経路はmetadata検証によりflatになりました。

正常bundleを保存した後、実際の`_atomic_write_bytes`に障害を注入しました。default/h=3/h=5で、Ω保存前・metadata保存前・manifest公開前の各失敗を確認し、障害が実際に到達した回数も記録しています。

| 条件 | 新loaderの検出 | 通常decide（h=1） | compute_distribution / h=3,5 blend |
|---|---|---|---|
| 正常bundle | metadata有効、alertなし | PASSED、gross≒2 | 正常cache利用 |
| Ω保存前に失敗 | 新μ・旧Ω、digest不一致 | FLAT、gross=0 | 新μ・旧Ωを利用、on-demand呼出し0回 |
| metadata/manifest保存前に失敗 | metadata=None、digest不一致 | FLAT、gross=0 | 新配列を利用、on-demand呼出し0回 |
| metadataなし再保存 | metadata=None | FLAT、gross=0 | 配列を通常利用 |
| manifest欠落 | commit manifest missing | FLAT、gross=0 | 配列を通常利用 |
| 読込途中で新版公開 | digest不一致、metadata=None | 別途上記の拒否経路を確認 | 低層loaderは非strictで配列を保持 |

混在h=3/h=5を入れるとblendのscoreは正常h1から最大約2.56677変化し、blendの返すalertは空でした。これは再現用score差であり、本番損益の推計ではありません。

**具体修正：**

1. 分布として無効なbundleは、非strictでも通常利用できる配列として返さない契約にします。例えばdigest不一致・manifest欠落では`(None, None, None, alerts)`、strictでは例外とし、3値wrapperでも無効状態を保持します。`[FATAL]`を付けるだけでは、alertを無視するblendが残ります。
2. 「digest整合」と「来歴が有効」の両方を分布利用の条件にします。metadata=Noneの互換経路を暗黙に許可しません。診断用に未検証配列を読む必要がある場合は、そのAPIを意思決定・baseline IR・学習用の取得経路から区別します。
3. `compute_distribution`、`apply_multi_horizon_blend`、Step 2のbaseline IR、研究の`_multi_horizon_scores`まで確認します。不正cacheは許可されたon-demandへ、不可能なら既存のflat/停止方針へ進めます。正常cache優先と正常計算の回帰は維持します。
4. 追加回帰は`load_gap_bundle`単体でmetadata=Noneになる確認だけで終えず、実3値wrapperと実利用側まで通します。default/3/5、保存障害3箇所、manifest/metadata欠落、並行読込みを対象にし、不整合なhorizonでscoreが生成・混合されないことを完了条件にします。

証跡：[reproductions.json](./reproductions.json)の`U01_bundle_boundaries`。再現コードは[probe_review.py](./probe_review.py)です。

**V02 / P2 — 注文状態の優先順位が部分失効を扱い切れていません（F01／R02）。**

対象：[Tachibana client.py:450](/Users/shonen/leadlag/src/leadlag/broker/tachibana/client.py:450)、[同:452](/Users/shonen/leadlag/src/leadlag/broker/tachibana/client.py:452)。仕様照合先：[立花証券API.md:885](/Users/shonen/leadlag/docs/api/立花証券API.md:885)。

取消完了コード7は数量判定より前へ移り、取消済み部分約定の旧再現は解消しました。しかし一部失効コード11等は数量判定の後です。`sOrderOrderSuryou=100`、`sYakuzyouSuryou=30`、`sOrderStatusCode=11`を実clientへ与えると、`PARTIALLY_FILLED`になります。コード11の失効状態へ到達せず、残量が約定し得るpendingとしてpoll対象に残ります。

同じ関数はコード5（訂正失敗）と8（取消失敗）を、約定量0なら`FAILED`へ変換します。これらは元注文の約定・取消完了を示すコードではありません。元注文まで失敗終了したと解釈してpollを終えるのは不適切です。原注文の状態・残数量が確定するまで、変更/取消要求の失敗と分けて扱う必要があります。

この確認は「全注文拒否が再び正常終了した」「未約定をFILLEDにした」という再発ではありません。全拒否は例外、通常のSUBMITTED→PARTIALLY_FILLED→FILLEDは最後までpoll、取消済み30株の収集も維持しています。残るのは変更要求の失敗と失効を含む状態分類です。

**具体修正：**

1. 仕様の状態コードを、原注文の終端状態、受付・約定途中、訂正/取消要求の結果に分けて明示的に整理します。失効等の終端状態を数量によるPARTIALLY_FILLED推定より先に判定します。
2. 失効を既存の`FAILED`へ写すか専用状態を追加するかは統一して決め、submit・close・poll・summary・CLIの全利用側を合わせます。どちらでも既に約定した30株は失わせず、詳細照会・建玉・余力・journalの収集を維持します。
3. コード5/8だけで原注文の残りが消えたと判定しません。確定していない原注文は照合対象へ残し、必要な再照会の結果またはdeadlineで状態を決めます。再送・追加注文を自動化する修正は不要です。
4. 回帰は、コード7/9/10/11/12と訂正失敗5・取消失敗8について、約定量0・部分・全量を組み合わせます。矛盾した数量/状態も成功と推定せず明示します。実clientの変換とsubmit/closeのpollをつなぎ、終端なら終了、原注文未確定なら追跡継続することを検査します。

証跡：`F01_F02_F20_brokers`。fixtureは実clientの詳細応答だけを置換し、外部APIは呼んでいません。

**V03 / P2 — common-input cacheのキーから列名が抜け、銘柄列の訂正後も旧計算を返します（F11）。**

対象：[blp_base.py:145](/Users/shonen/leadlag/src/leadlag/models/blp_base.py:145)、[同:158](/Users/shonen/leadlag/src/leadlag/models/blp_base.py:158)。

値・target・9:10価格のhash追加により、元の「数値2→100でもcache hit」は解消しました。一方、現在の`hash_pandas_object(df_exec, index=True)`は行の値を識別するもので、列ラベルを含みません。以前のキーにあった列名の識別がなくなっています。行の数値配列を変えず、`us_cc_XLB`と`us_cc_XLC`の列名だけを交換すると同じcache keyになりますが、実builderは銘柄名で入力を選ぶため計算上は別入力です。

まず小さい入力とbuilderの呼出し監視で、同じモデルではbuildが1回、訂正後も旧値0.02を返し、freshモデルでは0.04になることを確認しました。さらに**固定回帰データ4135行×122列と実`build_common_inputs`**でも検証し、cache再利用側とfresh側の`all_returns_raw`に最大絶対差 **0.0683854** が生じました。この後段の数値確認ではbuilderをモックにしていません。

条件は同じモデルインスタンスへ列の意味を変えた入力を再投入する場合です。日々の入力が同じschemaのまま数値だけ更新されるケースや、通常のhorizon切替が再び壊れたという主張ではありません。

**具体修正：**

1. DataFrameのcache identityに列名・列順・dtype、indexの意味、shapeと値のhashを含めます。少なくとも今回の列ラベル交換が同じキーにならないようにします。`p_910_df`にも同じ契約を適用します。
2. 共通のDataFrame fingerprintを使う場合は、必要なschema情報を省かないように設計します。canonicalな列順に正規化する方法でも、銘柄名による意味の違いがhashに残る必要があります。未知列・重複列の扱いも定義し、単にラベルを捨ててvaluesへ変換しません。
3. 回帰は「同じindex/shape/数値bufferで列ラベルだけ交換」を追加し、同じインスタンスの結果がfreshインスタンスと一致することを実builderで確認します。従来の数値訂正・target訂正・9:10価格訂正・horizon切替の回帰も維持します。

証跡：`F11_column_identity`、[schema_cache.json](./schema_cache.json)、[実builderの再現コード](./probe_schema_cache.py)。前回までのレビューでもschemaを含む契約の未了は言及していましたが、今回は実数値の差まで確定しました。

**V04 / P2 — VaRのoverlay版は固定されましたが、gap入力の版はkey計算とBTで変わり得ます（F06／R05の残存条件）。**

対象：[var_history.py:226](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:226)、[同:303](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:303)、[同:338](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:338)。

U02のoverlay修正は有効で、CURRENTをA→Bに切り替えても同じモデルobjectをBTへ渡すことを確認しました。しかしgapはfingerprintを先に計算し、BTには更新可能なDBパスを渡します。key計算後にgap生成・訂正処理がcommitすると、AのkeyへBの分布から計算した系列を保存できます。個々のμ/Ω/metadataがtransactionで揃っていることとは別の問題です。

一時SQLiteにAを保存し、実VaRがAのkeyを作った直後、cache照会の境界でBをcommitしました。実config loader・VaR関数・GapStore・SQLite return cacheを使用し、重いBT数値ループだけを読み込んだ版の識別値に置き換えています。BTが読んだ版はBなのに保存keyはAでした。

さらに接続が閉じ、WAL/SHMがないことを確認して、一時DBだけを元のA snapshotへ戻しました。同じkeyからBの識別値0.02が返り、BTは再実行されませんでした。Aの識別値は0.01です。これらは検証用マーカーで、実運用の収益率ではありません。本番DBの復元・編集はしていません。

**具体修正：**

1. risk計算で使うgap入力のsnapshot/版を、key計算前に選び、BTにも同じものを渡します。SQLiteならtransactionと整合したbackup/snapshotを作って、そのsnapshotの識別子とパスを使う方法があります。`.npy`ならimmutableなbundle版一覧を固定します。
2. live/研究のwriterが元パスを更新しても、進行中のBTが別版を読まないようにします。snapshotをコピーする場合、SQLiteの本体だけをファイルコピーしてWALを取り落とさないでください。
3. 短期対策として実行前後のversion比較で変更されたrunの保存・採用を拒否するなら、版がA→B→Aと戻る場合も検出できる更新世代を使います。mtime比較や「終わった時に同じ内容だった」だけをsnapshot保証としません。
4. 回帰はkey作成後とBT途中のgap commitを注入し、AのkeyへBの系列が入らないことを実cacheで確認します。変更前後のWAL hashが異なるという既存確認だけでは、実行中の入力固定を検証できません。overlay版固定の回帰も同時に維持します。

証跡：[var_gap_version.json](./var_gap_version.json)、[再現コード](./probe_var_gap_version.py)。

**元のF01〜F22の確認表**

「再現解消」は元の発生条件と今回記載の境界で確認した意味で、全ての市場データ・API応答・運用状態を保証するものではありません。「残存」は実害を再現したもの、「運用未了」は実artifact等の準備が必要なものです。

| ID | 判定 | 今回の確認・残る条件 |
|---|---|---|
| F01 注文状態 | 残存 | 全部約定・取消・全拒否・通常の部分約定は正しい。部分失効/訂正・取消失敗はV02 |
| F02 全拒否 | 再現解消 | 2注文とも拒否でaccepted=0、failed=2、OrderExecutionIncomplete。close CLIも2 |
| F03 risk stopと削減 | 再現解消 | 実損失閾値超過＋flat目標で削減経路を許可。削減/flatは許可、増加/反転/新規は停止 |
| F04 監査失敗 | 再現解消・範囲限定 | 同日signal/tradeの実リークFAILEDでgross=0、audit_failure=true。MH未来h=3、metadata欠落も拒否。h=1/3/5の未来target摂動はμ/Ω差0 |
| F05 過去日再利用 | 再現解消 | 指定日がdf_execにない実bridgeは例外。過去日のrunnerへ置換しない |
| F06 継承設定・VaR | 一部残存 | 実Step 2 mainはstrict loaderを使用し、本番v2と同じ設定・baseline IRパラメータ。VaRも継承loaderへ接続。実行中のgap版固定はV04 |
| F07 9:10価格 | 再現解消・範囲限定 | 偽APIの始値1000/現在値1050でcurrentだけを取得して1050を採用。cache分離・当日キーは既存unit test成功。実約定との照合は未実施 |
| F08 日米休日対応 | 再現解消 | 日本休場の08-11のUS+10%が08-12のJPへ対応、翌08-13は0%。休場paddingでも次の営業日を保持 |
| F09 strict前処理 | 再現解消 | 正・有限の80営業日rawでstrict前処理成功、signal日<取引日を確認 |
| F10 相関/horizon cache | 再現解消 | 入力値変更後はfreshモデルと一致。h1→h3とfresh h3のμ差・Ω差0。h1/3/5の未来摂動も差0 |
| F11 common-input cache | 一部残存 | 数値・target訂正は反映。列の意味を変える場合はV03 |
| F12 bundle atomic性 | 一部残存 | SQLite default/3/5の途中書込みをrollbackし旧版保持。読込中commitでも同一snapshot。`.npy`の一部利用側はV01 |
| F13 標準BTのML | 自動ロード確認／運用未了 | 一時的な正常artifactを設定した実BT入口からoverlayが渡る。実本番artifactでのweights照合は未了 |
| F14 flat日の決済費用 | 再現解消 | flat遷移は20.625→9.375bps、符号反転・同じweightも在庫フロー手計算と一致 |
| F15 初期wealth/DD | 再現解消 | −10%,0%のMDD=-10%。engineとplotも共通drawdown関数を使用し、既存回帰成功 |
| F16 対象日・複利 | 再現解消 | fallbackを含む4日でn=4、総収益=-8.2%、MDD=-10%、fallback率50%。主reportも全日を使用 |
| F17 DSR頻度 | 再現解消 | 年率化245日/252日の両方で、それぞれ同頻度の参照式と一致。245日では0.844025、252日では0.840624 |
| F18 A7/V1の扱い | 再現解消・用途限定 | 実A7 mainは明示フラグなしで停止、データ読込0回。現行V2のOOS証拠を新たに作ったわけではない |
| F19 ML来歴 | validator確認／運用未了 | null/NaT/空/配列/期間逆転/空hash/未来label等を拒否。legacy本番artifactは引き続き拒否、再生成は未了 |
| F20 fill収集 | 再現解消 | FILLED・取消済み部分約定も照会し数量を保持。detail失敗や初回ログ障害でも後続照合を試行し失敗を伝播 |
| F21 PIT multiplier | 再現解消 | 0.25設定で実multiplier=0.25、fallback_flag=true。分布取得不能のflatとは区別 |
| F22 regression fixture | 再現解消 | 固定4135行入力・2026-08-14で非zero baselineを検証。別途同一processでpytestを実行し、終了後のmacro関数identity復元も確認 |

**後続レビューのR/S/T/Uに対する再確認**

| 指摘群 | 今回の結果 |
|---|---|
| R01 / S05 provisional・休日padding | strict/非strictと固定unit回帰成功。未来USの同日JP入力への接続を拒否し、JP休場padding後の有効日を保持 |
| R02 / S02 / T04 / T05 注文・照合 | 通常の部分約定は3回の状態照会でFILLEDまで到達。取消後30株保持、部分失敗後のfills/positions/wallet/journal継続。失効等の追加状態はV02 |
| R03 / S01 / T03 分布の来歴 | 主decideの未来h3・欠落来歴をflat化。`.npy`互換利用側の残存はV01 |
| R04 SQLite読込み | default/3/5で更新割込み後も読込中の旧版μ/Ω/metadataが揃い、次回は新版 |
| R05 / U02 VaRの版 | WAL更新をfingerprintが検出。overlayのA→B切替でもA objectをBTへ渡し、Aへ戻したcacheもA系列。gap側はV04 |
| R06 / S04 / T01 / T02 ML artifact | in-sampleをMLなしの成績へ変換せず拒否。save/load/applyの来歴検証、digest・内外metadata一致、公開失敗時の旧CURRENT維持を確認 |
| R07 DD | 指標・engine・plotの共通関数への接続と初期損失の回帰成功 |
| R08 テスト汚染 | regression単独の同一processで、pytest終了後のglobal macro関数復元を確認 |
| S03 close CLI | 全拒否・detail照合失敗ともCLI=2。休日skipを無効にし実close処理到達を確認 |
| U01 `.npy`再保存 | 主decideで不正bundleをflat化。digest照合は正常に検出するが、3値APIの利用側にV01 |
| U03 検証ツール | 正常版はexit=0、active model/active directory欠落はexit=1、実loaderも拒否 |
| U04 移行確認 | 正常artifact内部をlegacyと誤検出せず、main exit=0 |

**段階A〜Cの完了判定と残る運用準備**

段階AにはV02、段階BにはV01/V03/V04が残ります。段階CのBT・統計の元の再現は改善していますが、来歴を保証した分布入力・risk cacheとの対応と、実本番ML artifactの準備が完了していません。したがってA〜C全体の完了・リリース可とは判定しません。

本番設定は`ml_overlay_enabled=true`、`model_dir=models/ml_order_overlay/phase2_8`で、現在のartifactはlegacyのため実`ProductionRunner`が拒否します。この安全側の拒否を新規不具合として数えていません。検証可能なデータによるversioned artifact再生成と、本番/BTで同じ入力・同じartifactを用いるweights照合は未実施です。

固定regressionのmodel gross=1.5、net≒0で非flatを維持しています。設定のside_leverage=1.5を適用した対応値は実効gross=2.25、net≒0で、設定のgross上限3.0・net上限0.05以内です。別の合成正常bundleはmodel gross≒2、実効gross≒3です。これらはモデル出力の確認で、実際の口数丸め・約定後exposureの証明ではありません。

解決済みコストは片道slippage=5bps、long持越し0.75/short0.5、金利年2.5%、貸株年1.15%、reverse fee=2bps、side leverage=1.5です。精密な実約定PnL、新たなOOS収益実験、終端在庫を含む口数ベース台帳の再設計は今回行っていません。baseline期間2010–2014のコード、strictly historicalな相関窓、future-target摂動を照合しましたが、macro/ADR等の全外部系列の公表時刻を網羅的に証明したものではありません。

ADR・ARCHITECTURE・技術仕様書の記述と実装を照合しました。gapの「利用まで一貫したbundle」とcacheの「実際に使った入力版」はV01/V04を含めて成立させる必要があります。`docs/refactor_roadmap.md`の部分完了・未完了も確認し、過去のPhase状態や段階D〜Fを今回の修正へ追加していません。

**同じ修正の往復を減らすための受入条件**

今回の残存4件は、いずれも関数単体での修正と下流の利用条件の間にあります。次回はテスト件数だけで完了を判断せず、以下を修正単位にしてください。

- 分布：producer→4値loader→3値wrapper→通常decide / compute_distribution / baseline IR / 学習・研究の利用側まで、同じ不正入力で拒否を確認する。
- 注文：API応答→中立状態→submit/closeのpoll→summary→journal→CLIまで、部分約定量を保持したまま状態遷移を確認する。
- cache：値・schema・horizon・モデル版・gap版について、「変更したら別計算」「同じrun中は同じ入力」をそれぞれ確認する。
- 完了判定：このF01〜F22表の残存条件を閉じ、正常経路の非flat回帰も残す。レビュー用の再現コードを必要な恒久テストへ移す際は、古い関数名や旧APIのmockをそのまま複写せず、故障注入の到達をassertする。

**検証と証跡**

| 検証 | 結果 |
|---|---|
| 全`tests/` | 620 passed、17 warnings、378.67秒、10workers。watchdog exit=0、379.7秒、全体期限1800秒：[pytest_full.log](./pytest_full.log) |
| Ruff | 本番・tests・変更された研究共通コード/実験script/toolsの指定範囲で成功：[ruff.log](./ruff.log) |
| mypy | 本番120 source filesで成功：[mypy.log](./mypy.log) |
| compileall | src/leadlag・tests・tools・scripts・src/researchで成功：[compileall.stderr](./compileall.stderr) |
| import-linter | 154 files、381 dependencies、4 contracts kept：[import_linter.log](./import_linter.log) |
| その他 | `git diff --check`、変更batchの`bash -n`成功 |

17 warningsは研究テストのゼロ除算・定数列相関です。研究コード全体のRuff/mypyを成功したという意味ではありません。プローブで発生する意図的なOSErrorやrisk-stopのログは、失敗条件の検証証跡です。

主な再現コマンドは以下です。いずれもリポジトリルートから実行し、プロセスグループ全体の停止期限を持ちます。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 240 .venv/bin/python reports/20260914_stage_abc_round5_review/probe_review.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260914_stage_abc_round5_review/probe_contracts.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260914_stage_abc_round5_review/probe_schema_cache.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260914_stage_abc_round5_review/probe_var_gap_version.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_model.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260913_stage_abc_rereview/probe_remaining.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n auto --tb=short -q
```

追加の証跡は[contracts.json](./contracts.json)、[model_probes.log](./model_probes.log)、[prior_probes.log](./prior_probes.log)、[regression_isolation.log](./regression_isolation.log)です。旧probeは当時のAPIを前提とするため、今回の[probe_review.py](./probe_review.py)では必要な定義を再利用しつつ、書込障害の注入点や期待する拒否の扱いを更新しています。DSRの旧参照値も年率化係数を明示して245日/252日を別々に検証しました。

開始・終了時のhashは[source_snapshot.json](./source_snapshot.json)と[review_manifest.json](./review_manifest.json)、指摘とF01〜F22の機械可読一覧は[review_summary.json](./review_summary.json)、リンク・証跡・判定の整合は[validation.json](./validation.json)を参照してください。
