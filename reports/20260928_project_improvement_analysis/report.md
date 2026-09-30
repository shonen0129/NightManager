**プロジェクト改善分析 — 2026-09-28、現在の作業ツリーを対象**

最優先は、実運用対象の9:10→大引けリターン、予測時点の入力、実約定と費用を同じ実験・同じ日付で結び付けることである。その基盤に、設定・指標・認証情報保存の具体的な不備を修正する。その後に、既存モデルのどの部分が利益を生み、どの部分が複雑さだけを増やしているかを測る。この順序が、現時点で最も成果につながりやすいと判断する。

この結論は「モデルに改善余地がない」という意味ではない。現状は、予測精度の小さな改善より、測定対象と執行条件の差の方が大きく、研究上の順位が実費を入れると変わり得る。新しい特徴量やモデルの優劣を判断する前提を整えること自体が、精度改善の一部である。

**調査範囲と証拠の区別。** 現行コード、本番設定の継承解決結果、構造ADR・ロードマップ、最近のJPX33・収益改善・実約定・ML・VaRレポート、テストとCI定義を確認した。多数の未commit変更を含む現在の作業ツリーを読んだ。初回分析時には、新しいバックテスト、全テスト、実API・発注・scheduler操作を実施せず、既存ソース・設定・運用データも変更しなかった。その後の即時修正と検証は末尾の追補に記録する。

以下の「再現」は今回の小さなオフライン検証、「現コード」は呼出し経路から確認した仕様、「既存結果」は過去レポートの数値、「仮説」は未検証の改善案を表す。これは改善方針の分析であり、全リリース条件を監査したPASS判定ではない。実費込みの収益性と新規改善の本番採用判断は、必要な証拠が揃うまで保留が妥当である。

**MECEの整理。** 改善対象を、それぞれが答える問いで7領域に分ける。たとえばquote取得時刻はデータ、板を見た注文数量は執行、実口座損失による停止はリスク、比較の有意性は実験評価に置く。領域間には依存があるが、同じ作業を重複して数えない。

| 領域 | 答える問い | 主要な改善成果 |
|---|---|---|
| A データ・ターゲット | その時点で何を知り、何を予測したか | 実観測時刻、価格の種類、PIT、調整基準を持つ入力 |
| B 予測モデル | 将来リターン・順位・リスクをどれだけ予測できるか | 同一targetで測れるμ・Ω・MLの増分価値 |
| C 配分・執行・会計 | 予測を売買して、いくら利益が残ったか | 実在庫・板・貸株・費用を反映した注文と損益 |
| D リスク | どれだけ保有でき、いつ止めるか | モデルリスクと実口座損失を別々に制御 |
| E 実験・統計評価 | 改善と判断できる証拠があるか | 共通指標、比較日付、全試行、未使用評価期間 |
| F ソフトウェア構造・品質 | 意図した計算を安全に再現できるか | 厳密な設定境界、秘密情報を除いた記録、現行経路の回帰 |
| G 運用・研究管理 | 毎日回り、結果と採用状態を追跡できるか | 遅延・欠損・照合・モデル版・採用根拠の継続監視 |

**現状の基盤は維持する。** V2同期経路、型付きDecisionInputs、共通モデル構築、bundleの来歴・transaction、注文intentの事前保存、未解決注文の再送防止、プロセス群の期限、数値・リーク監査、production wheelからのresearch分離、import境界、lockfile、全testsを対象とするCIは既に存在する。全面的な作り直し、非同期化、別の巨大Runnerの導入を主改善にしない。古い監査の指摘には修正済みのものが多く、現在のコードを優先した。

現行設定を `load_config_from_yaml(..., strict=True)` で解決すると、ML有効、h=1/3/5の重み0.8/0.1/0.1、on-demand有効、監査失敗時fallback有効、shadow cache/on-demand比較は無効である。片道slippageは5bps、side leverageは1.5、モデルgross上限2.0に対して実効gross上限はrisk設定3.0。持越し比率はlong 0.75 / short 0.50。年率金利2.5%、貸株1.15%、reverse 2bpsも含む。[解決済み公開設定と再現結果](probe_results.json)に保存した。認証設定は保存していない。

**A. データ・ターゲットを改善する。**

**A1 — 9:10 quote、5分足近似、始値代替を別のデータ契約にする。優先度：最優先。** 長期感応度比較2,779日の9:10価格proxy被覆は銘柄日の3.13%、17銘柄が揃う日は8日。これは実際の9:10→大引け戦略を完全に観測した2,779日ではない。5分足の09:10行から `(High+Low)/2` またはCloseを取り、欠損時は始値へ代替する実装がある。さらに `resolve_execution_prices` は値があれば `observed_0910` と表示するため、実quoteとbar由来proxyをこの欄だけで区別できない。[JPX33知見まとめ](../20260927_jpx33_knowledge_summary.md)、[intraday_inputs.py](../../src/leadlag/data/intraday_inputs.py) 51–81、172–216行。

改善案は、銘柄日ごとに `quote_mid_at_cutoff / bar_proxy / daily_open_fallback / missing` を持たせ、target種別・価格調整基準と一緒に保存すること。長期open→closeは別の研究系列として保持し、正式な9:10比較に混ぜない。既存データを埋め直して被覆率を上げたことにしない。バーの時刻が始端か終端かをprovider仕様と照合し、始端09:10のHigh/Lowなら確定は09:10より後になることをavailabilityへ反映する。今回、全barの仕様を認証したわけではないので、全バックテストのリークを断定しない。

受入条件は、全評価日の価格種別・欠損・flatが追跡できること、正式比較では始値代替が混入しないこと、欠損により採用日を変えた場合も全営業日基準の成績・欠損率を併記すること。完全観測日の成績だけを全戦略の主成績にしない。

**A2 — 宣言された09:10から実観測時刻へ移す。優先度：最優先。** 当日価格cacheは銘柄と価格だけを保存する。live bridgeはcutoffを09:10に固定し、入力の `observed_at` にも09:00/09:10を組み立てて渡す。この値はproviderの実配信・受信時刻の証明ではない。日付が同じ09:20の値を09:10観測と区別できない経路が残る。[session_cache.py](../../src/leadlag/broker/tachibana/session_cache.py) 172–209行、[v2_bridge.py](../../src/leadlag/execution/v2_bridge.py) 66–68、473–515行。

改善案は、exchange/event時刻、providerで利用可能になった時刻、受信時刻、計算cutoff、raw payload hashを区別し、銘柄別に保存すること。最終値から過去のavailable_atを推測して補完しない。歴史入力の `session_boundary_contract` は想定時刻として明示し、実測時刻とは区別する。09:10時点のsnapshotを凍結してgap・decision・shadow・後日の再生が参照する。同日再実行で新しい価格へ置き換わらないことも検査する。

受入条件は、cutoff後の観測と古いcacheを拒否または明示的な非適格状態にすること、当日未知target・未来データを変えても当日予測が変わらないこと、cache/on-demandで同じ入力・モデル版を比較できること。

**A3 — 価格調整と分類の来歴を一般化する。優先度：次段階。** 現行の分足loaderには1629.Tの分割補正が銘柄・日付・倍率を指定して実装されている。これは修正済みの事象を再度バグとする指摘ではなく、次の分割にも同じ品質を保つための改善である。[market_data_cache.py](../../src/leadlag/data/market_data_cache.py) 36–65行。

corporate actionをデータとして版管理し、raw価格、分割調整価格、分配金を含むリターン、実際の口数・約定価格を混同しない。日足と分足の調整係数一致、二重補正防止、単元・保有口数の整合をテストする。JPX33の再開時には、現在構成・総時価総額の比率ではなく、時点内構成と対象指数を再現する重みを用意する。JPXはTOPIXを浮動株調整時価総額加重と説明し、TOPIX-17もTOPIXに基づく指数としている。総時価総額比と対象指数の比率を同一視できない。[JPX TOPIX](https://www.jpx.co.jp/english/markets/indices/topix/)、[JPX指数一覧](https://www.jpx.co.jp/english/markets/indices/line-up/)。

**B. 予測モデルは、何が効いているかを分解して改善する。**

**B1 — 精度を単一のRank ICやMAEに集約しない。優先度：次段階。** 同じ9:10→大引けtargetで、日次横断Rank IC、上位/下位群の実現差、予測値の校正、方向一致、共分散のリスク予測、最後に実費込みのポートフォリオ価値を順に見る。MAE改善だけでは、ロング/ショートの順位選択や利益が改善したとは限らない。日付でまとまった誤差を扱い、17銘柄日を全て独立標本と数えない。

JPX33直接予測では、OHLC CSV版でMAEは約0.76〜0.77bp改善したが、Rank IC差は約−0.0038〜−0.0042で改善の証拠がない。proxyとETFの写像品質にも制約がある。現行17次元を暫定baselineとし、データが整うまでは33次元化を主開発テーマにしない。[JPX33知見まとめ](../20260927_jpx33_knowledge_summary.md)。

具体的な分析単位は、銘柄、long/short、時間帯、予測分位、事前に定義した市場状態、データ品質である。どこで誤差・コストが増えるかを診断してから、局所的な仮説を立てる。事後に成績のよい期間を発見して新しい採用OOSとは扱わない。

**B2 — MLの役割を相対配分と取引判断に分ける。優先度：次段階。** 現行回帰モードの `p_trade` は校正済みの勝率ではなく、回帰出力をtarget標準偏差でsigmoid変換した配分倍率である。さらにlong/shortを各側で基準grossへ正規化するため、一律倍率による「自信度低下」は通常は相殺される。現在のモデルは主に各側の銘柄間配分を変えている。[ml_overlay_features.py](../../src/leadlag/models/ml_overlay_features.py) 164–178行、[ml_overlay_inference.py](../../src/leadlag/models/ml_overlay_inference.py) 103–118行。

学習targetも `sign(score) × realized − 固定往復コスト` であり、現在在庫から注文する増分利益の直接教師ではない。[ml_overlay_training.py](../../src/research/experiments/ml_overlay_training.py) 170–181行。改善案は、まず名称と出力契約を明確にして過剰な確率解釈を防ぐこと。研究では、既存の相対配分overlayと、校正済み期待収益・費用・在庫から作る注文価値を別々に評価する。後者は実在庫・quote・費用が整ってから、特徴量、label horizon、校正学習、選択期間を固定する。

単純なOLS校正や既存の注文コストゲートを新しい有望案として再提案しない。既知期間の校正主候補はMAE差−0.08bpで区間がゼロを跨ぎ、注文ゲートもbaselineに劣後した。精度の改善は仮説であり、採用済みではない。[順序11](../20260923_profitability_order_11/report.md)、[順序12](../20260923_profitability_order_12/report.md)。

**B3 — 既存部品の削減とMHの作用を優先診断する。優先度：データ整備後。** 既知368日のablationではMH無効化でSharpeが5.164から0.809へ低下した一方、ML・copula・minvarを外すと一部の指標は改善した。ただし、全ての採択条件を満たす削除案はなかった。この表だけで機能を削除しない。[順序6](../20260923_profitability_order_6/report.md)。

まずh=1/3/5ごとの入力窓、ラベル確定時刻、予測尺度、共分散尺度、ブレンド後の銘柄順位を同じsnapshotで保存する。MHの大きな差が、追加情報、リスク推定、順位変化のどこから来るかを分解する。μの精度とΩの予測リスクを別に採点し、事前固定した部品除去比較をforwardでも追う。モデル追加による複雑化より、効果の弱い処理の整理を候補にするが、実費・DD・fallbackを含むOOS条件を通す。

**C. 配分・執行・会計で実際に残る利益を測る。**

**C1 — 全約定と在庫・現金・費用を照合する。優先度：最優先。** 公式CSV339件/4,130株の整理は進んでいるが、ローカル記録と価格まで一致した範囲は7グループ8株である。全件の実slippageやnet PnLが確定した状態ではない。[実約定レビュー](../20260927_actual_execution_review/report.md)。

既存order intent→broker order ID→fill ID→建玉lot→cash/feeの対応を完成させる。重複取込、部分約定、訂正、取消、持越し、最終清算、週末金利、分配・権利処理を同じ台帳で扱い、unknown費用を0としない。実約定価格を使う損益から同じslippageを二重控除しない。成功は、未照合件数と金額が説明でき、前日資産・入出金・価格変化・費用・当日資産が一致することとする。

**C2 — コスト感度の主語を正す。優先度：最優先。** 既知368日・保存済みウェイトを固定したML on/offの再価格付けは次の通りである。これは9月23日時点の比較に基づく診断であり、その後の2025年末cutoffで再学習した現行artifactの独立成績ではない。[コスト感度報告](../20260927_ml_overlay_cost_sensitivity/report.md)。

| 片道slippage | ML on Net Sharpe | ML off Net Sharpe | 読み取れること |
|---|---:|---:|---|
| 5bps | 5.164 | 5.180 | ML onの単純net合計は増えるがSharpeは低下 |
| 10bps | 3.115 | 3.089 | 絶対成績は大きく低下、増分平均差CIは0を跨ぐ |
| 20bps | −0.971 | −1.084 | 両系列ともこの仮定では平均netが負 |

同報告の約67bpsは「ML on−off差分」の損益分岐であり、戦略全体の耐コスト性ではない。順序8の数値から、grossとウェイト・他費用を固定すると、**日次netリターンの単純合計が0になる片道slippageは約17.62bps**と計算できる。これは複利資産の損益分岐や、実impact・非約定・整数口数を含む許容コストではない。主な改善は、固定5bpsをより楽観・悲観な数字へ替えることではなく、実績で銘柄・売買側・数量・時刻別の費用分布を測ることである。

**C3 — 板・貸株・口数を注文へ接続する。優先度：次段階。** 板spread、推定slippage、depth、short可否の部品は存在するが、`apply_hard_rules` と `replace_unavailable_short` はsrc/testsで定義以外の呼出しを確認できない。本番新規注文はMARKETと株数に基づく分割である。[execution_constraints.py](../../src/leadlag/execution/microstructure/execution_constraints.py)、[broker_ops.py](../../src/leadlag/execution/broker_ops.py) 685–704行。

既存部品をそのまま有効にする前に、板欠損・計算例外時の扱いと貸株可否の鮮度を決め、shadowで判定差を保存する。現部品は板欠損・計算例外時に制約を適用しないため、接続だけで完了ではない。片側注文を除外したら両側の数量と中立性を再計算する。単元丸め後のnet調整は既存なので、改善対象は実fill後と部分約定中の制約、数量別の容量、実際の貸株制約である。

受入条件は、AUMごとの理論notional表に加え、注文/板金額、fill率、実現slippage、借株可能量、丸めと部分約定後のgross/netを説明できること。新optimizerの導入は必要性を測ってからにする。一律L1 turnover capは既に棄却されており、同じ案を再探索しない。[no-trade実験](../20260923_turnover_aware_no_trade/report.md)。

**C4 — 日中と持越しを別の投資判断として検証する。優先度：次段階。** 既存診断のovernight carry Sharpeは0.125、no-carry反実仮想のnet Sharpeは5.931、現行は5.164。ただし約定可能性・価格被覆に制約があり、carryを0にすれば改善すると確定できない。[順序5](../20260923_profitability_order_5/report.md)。

日中のリードラグα、引け後→翌寄付、寄付→9:10の在庫損益、費用節約を分ける。連休と急変、追加往復売買によるコストを含め、持越し比率の変更を少数の事前固定仮説として評価する。新しい比率を試すなら±感度と全試行補正を行い、真のforwardで確認する。

**D. モデルのリスクと口座の損失を分けて管理する。**

**D1 — 実口座PnLの損失停止を接続する。優先度：最優先。** 現在のDailyLoss/MonthlyLossは `hist_daily_returns` の最終日・最終月を使う。live bridgeから渡される系列はモデルreplay履歴であり、実口座の当日損失を直接見ているわけではない。調べたsrc/tools/scriptsでは、別の実口座PnL停止経路は確認できなかった。[risk.py](../../src/leadlag/core/risk.py) 152–183行、[v2_bridge.py](../../src/leadlag/execution/v2_bridge.py) 668–684行。

既存の実現・評価PnLレポートを材料に、前日評価差と入出金、全費用を調整した口座リターンを作り、モデルVaRとは別の損失停止へ渡す。[daily_pnl_report.py](../../src/leadlag/reporting/daily_pnl_report.py) 342–378行の集計をそのまま日次リターンと見なさず、持越し評価損益の差分と戦略/口座の帰属を定義する。

受入テストは「replayはプラス、実口座は損失停止超過」の場合に新規リスク増加を止められること。停止時にも安全な縮小・決済を許可する既存方針を維持する。実口座データ欠損時の扱いも明示する。

**D2 — 市場中立を金額とβに分ける。優先度：次段階。** モデルnet≈0でもβ中立は保証されない。順序8の `Σw×jp_beta` proxyは絶対値p95約0.536、最大約1.047だった。これはgap βの診断で、日中の実現βを直接証明しない。[順序8](../20260923_profitability_order_8/report.md)。

モデルraw、side leverage後、整数配分後、実fill後のgross/netを同じ表にし、資産額の分母も統一する。共通 `max_gross_exposure` をrawとallocation両方で解釈する箇所は、model limitとeffective limitを別の名前・型で表す。日中・持越しの因子βと集中度を別に測り、必要なら歴史データだけで推定したβ制約を研究する。即座にβ=0へ強制するとαも削る可能性があるため、採択条件と感度検証を先に決める。

**D3 — VaR/ESの不確実性を示す。優先度：次段階。** 履歴不足は既に解消されている。最新報告の250標本ではVaR99=3.053%、ES99=4.591%で、停止基準3%/4%を超え、tailは3件。今回そのリスク履歴を再計算してはいない。[VaR履歴報告](../20260927_ml_overlay_var_history/report.md)。

改善は閾値緩和ではなく、tail日の損益・因子・価格source・費用の説明と、長い期間や異なる市場状態のstress、推定区間の併記である。停止原因を「履歴不足」「計算失敗」「数値上の経済的リスク超過」に分ける。side leverageやcarry縮小の効果は同じ費用モデルで別研究として検証する。

**E. 実験評価を一つの仕様に揃える。**

**E1 — 指標の重複実装を解消する。優先度：直ちに修正できる。** 現行 `experiment_utils._extract_metrics` は245日年率、順序5/6の独自集計は252日年率。Stage5の同じ日次系列は252日でSharpe 5.163636、245日で5.091414となる。245か252の選択そのものより、同名指標の定義が記録先で変わることが問題である。[experiment_utils.py](../../src/research/experiment_utils.py) 33–81行、[Stage6 script](../../src/research/scripts/experiments/experiment_stage6_overlay_ablation_20260923.py) 129–145、308–326行。

今回の再現で、Series `[0.01, NaN, −0.02, 0.03, 0.005]` はhelperのSharpe=0、観測数=5、保存returns数=4、MDDとtotal returnはNaNになった。欠損を成績0と表示し、DSRの標本数と系列を不一致にする。Stage6の `_stats([-0.1, 0])` はMDD=0、初期wealth=1を含む正本は−10%である。既存レポート全体のDDが何%誤っているかまで今回再計算したわけではない。

既存の [MetricsSpecとdrawdown関数](../../src/leadlag/reporting/metrics.py) を利用し、研究helper、日次報告、registry、DSRへ同じ仕様・同じ日次系列を渡す。primary指標はflat日を含む。NaN/Inf、日付重複・欠落は黙って成績へ変換せず、評価不能や明示的な除外理由にする。年率係数、頻度、target、費用、複利/単純合計、観測数、fallback分類をschemaとして保存する。

受入条件は、同一系列のreport/registry/DSR入力が一致すること、初日損失・全flat・欠損・空系列の定義が明確なこと。活性日の補助指標と全営業日の主指標を混ぜない。これは現在のactive fileに最も直接的な改善である。

**E2 — 訂正レコードと試行集合を管理する。優先度：直ちに整備できる。** 現在のhelperは明示 `trials` を保持し、DSR本体も年率から日次へ換算済みである。ここは過去の不具合を再指摘しない。一方、指定がなければ同名レコード数に依存するため、別名の類似候補や過去探索を自動的に数えられない。[experiment_utils.py](../../src/research/experiment_utils.py) 128–133行、[experiment_registry.py](../../src/leadlag/experiment_registry.py) 125行以降。

`study_id` と候補集合、baseline、全日次系列、選択基準、見た評価期間を記録する。独立試行数を不明なまま1とせず、試行の依存と未登録探索を含む範囲を明示する。DSRは選択バイアス・非正規性を扱う手法であり、入力誤りや未知の探索履歴を消すものではない。[Bailey & López de Prado, DSR原論文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。

長期感応度reportは、旧registryのRank ICがopen→09:10に対する値だったと明示している。訂正された本文と古いレコードが別の意味のまま残り得る。[長期感応度報告](../20260924_sensitivity_pipeline_audit_long/report.md) 72行。append-onlyを維持したまま `correction_of / supersedes / metric_schema_version` を持つ訂正レコードを追加し、通常の集計では「現在有効な結果」を読むviewを用意する。過去記録は削除しない。

**E3 — 未使用評価を守り、比較の検出力を設計する。優先度：最優先で開始し継続。** 2015〜2026年や既知368日を何度も見た後で、同じ期間をfresh OOSとは呼べない。歴史再生は診断に使い、forward候補・artifact・指標・判定時点を固定する。既存ML仕様の250完全paired日、感応度仕様の252営業日を継承し、都合のよい途中結果で採用条件を変えない。[ML forward報告](../20260927_ml_overlay_forward_shadow/report.md)、[感応度forward仕様](../../docs/decisions/2026-09-24-sensitivity-audit-forward-protocol.md)。

所定の日数を集めれば必ず改善を識別できるわけではない。検出したい実費込み改善幅とペア差の分散・自己相関から、必要標本と検出力を設計する。候補とbaselineは同じ日付で比較し、paired block bootstrapで平均差だけでなく主指標のSharpe差も評価する。purge/embargoはh=1/3/5のラベル終了時刻から設計する。新パラメータには±感度と試行補正を付ける。

**F. 構造面では境界の厳密さと再現性を改善する。**

**F1 — 認証情報を研究記録から除く。優先度：直ちに修正できる。** helperは `AppConfig.model_dump(mode='json')` 全体をparametersへ保存する。AppConfigにはbrokerのtoken/passwordが含まれ得る。実秘密値を使わず、ダミーtoken/passwordを渡して `/tmp` の一時registryへ保存したところ、3種類の値が平文のまま含まれた。[experiment_utils.py](../../src/research/experiment_utils.py) 116–121行、[experiment_registry.py](../../src/leadlag/experiment_registry.py) 221–224行。

実際の漏洩を確認したという主張ではない。保存経路の存在を再現した。改善は、研究に必要なv2/risk/費用・データ版・コード版だけをallowlistで保存する `ExperimentConfigSnapshot` を作ること。broker認証を引数へ渡す必要自体を減らす。SecretStrやログの伏字だけに依存せず、dict経路でも秘密項目がserializationされないことを合成値でテストする。既存記録の点検が必要なら、秘密値を表示・報告へ転載しない方法で実施する。

**F2 — 設定を誤ったまま実行できないようにする。優先度：直ちに修正できる。** `strict=True` でもnestedの `risk.var_stpo`、`ml_order_overlay.enabeld`、`costs.slippage_bps_per_sdie` は例外にならず、既定のvar_stop=3%、ML無効、slippage=5bpsで進む。明示した存在しないYAMLパスも既定値で成功する。今回合成設定で再現した。[execution/config.py](../../src/leadlag/execution/config.py) 101、116–122、272–279行、[schemas.py](../../src/leadlag/config/schemas.py) 416–423、593–602行。

正規化前に全階層の未知キーを検査し、alias衝突を検出し、明示pathの不存在は例外にする。暗黙default利用と明示configロードを別のAPI契約にする。resolved configの各値についてYAML継承元、env/CLI上書きの出所を表示できると原因調査も容易になる。認証値は表示しない。受入条件は誤字・欠落・衝突で停止し、正しい現行production.yamlの意味を変えないこと。現在の本番設定そのものが誤字で誤動作しているとは断定していない。

**F3 — 現行の型付き入口を回帰基準に加える。優先度：次段階。** 固定日付の既存golden snapshotは残す。一方、そのcaptureは `ProductionV2Model.decide(trade_date=..., gap_input_dir=...)` を使い、型付きadapterを意図的に通していない。コメントではplatformで変わるfingerprintを理由に挙げる。[test_v2_baseline.py](../../tests/regression/test_v2_baseline.py) 49–63、110–120行。

これを現行live/BTの全契約をカバーするテストとは扱えない。ProductionRunner→DecisionInputs経路に、固定quote、macro/ADR、PIT、ML artifact、h=1/3/5 bundleを揃えた追加fixtureを作る。cache成功、cache不採用→on-demand、入力不足→flat、監査失敗、未来target摂動を独立に検証する。入力のdtype・byte order・NaN表現などcanonical serializationを決め、platform差をhash検査の省略で吸収しない。既存unitの境界テストを全て作り直す必要はない。

CIは存在し、全testsを対象にしている。改善対象は「CIを新設」ではなく、現行入口のgolden検証と、共有研究コードへの検査範囲の拡張である。現在researchのRuffは主に学習入口、mypyはproductionが対象なので、experiment_utils、共通統計、order_economicsなど継続利用する研究部品を優先してCIへ加える。過去の全研究スクリプトの一括整形を先行させない。[CI定義](../../.github/workflows/ci.yml) 39–49、72–75行。

**F4 — 未追跡コードまで再現用hashへ含める。優先度：次段階。** runtime manifestのdirty hashはgit statusとtracked diffを材料にする。未追跡ファイルは名前が同じなら内容変更がhashに反映されない。 `/tmp` の新規git repositoryで未追跡Pythonの内容を変更し、dirty_diff_hashが変わらないことを再現した。[runtime_manifest.py](../../src/leadlag/execution/runtime_manifest.py) 57–81行。

再現に必要なコード・設定・artifactをallowlist化して内容hashを保存する。実行時に参照した未追跡ファイルも含めるが、秘密ファイルや全データを無差別にbundleしない。ユーザーの作業を強制commitさせず、run専用の不変snapshotを作る方法もある。受入条件は、実行コードの内容が変わればrun identityが変わり、そのidentityから環境と入力を再構成できること。

**G. 運用と研究の継続管理を改善する。**

**G1 — 09:10のcritical pathを短くし、所要時間を観測する。優先度：最優先。** 現バッチはquote/板capture→gap計算→decisionの順である。capture windowは既定30秒で、さらにgap処理とrisk履歴が続く。shadowのML off計算も発注前に同期実行される。例外処理で本番継続できても、遅延の影響は残る。[run_decision_v2.sh](../../scripts/batch/run_decision_v2.sh) 68–116行、[v2_bridge.py](../../src/leadlag/execution/v2_bridge.py) 521行以降。

まずquote取得、入力凍結、分布生成、モデル、shadow、VaR、注文受付、fillの時刻とp50/p95を測る。歴史だけで決まる処理とVaR履歴は前夜・寄付前にprewarmし、09:10の差分計算へ分ける。snapshot永続化後のshadow処理を発注経路外へ移す場合も入力hash一致を維持する。新しい並行発注経路を作る必要はない。診断処理の停止・lock待ちが本番執行期限を超えないことが受入条件である。

**G2 — forward記録を日末評価まで完成させる。優先度：最優先で開始し継続。** ML on/offの同一入力記録と板captureは実装済みだが、現forwardレコードは予測・weights保存までで、実現収益の付与が残る。[forward報告](../20260927_ml_overlay_forward_shadow/report.md)。

日末に実現target、candidateの実fill・費用、baselineの同資本・同単元・同じ観測板による反実仮想を結合する。baselineを実約定と呼ばず、推定費用の区間を表示する。非稼働・欠損・停止日を含む全営業日の台帳から、paired net差、Sharpe差、DD、turnover、fallback、overlay skipを出す。発注がrisk stopで0の場合も予測記録を残し、shadow予測とlive発注の成功を別に測る。

**G3 — 技術的有効性、研究の採用根拠、実稼働状態を分ける。優先度：次段階。** 現artifactは利用者指示により公開され、promotion記録は `operator_override=true`、元の数値gate不合格、forward label 0と記している。利用者が切替を指示した事実と、独立OOSで有効性が確認された事実は別である。[PROMOTION.json](../../models/ml_order_overlay/production_20260923/PROMOTION.json)。

artifactを読めるか、時点制約を満たすか、統計・実費の採用基準を満たすか、運用でactiveか、現在risk stopかを別欄に表示する。既存override記録を消さず、再評価日・必要証拠・rollback対象を追えるようにする。新しいpermission workflowを先に増やすことが目的ではない。

日次運用の成功もexit codeだけでなく、定刻入力、判断、注文状態、建玉・cash照合、日末targetの連結までで判定する。cache/on-demand/flat、PIT減額、ML skip、risk stop、注文拒否、未約定を分けた率を出す。ローカルstate store・gap store・raw観測・artifactについて、バックアップの存在だけでなく別場所へ復元して整合するかも確認課題とする。今回、外部バックアップや復元運用の有無を網羅確認したわけではない。

ロードマップは過去記録と現行残件を既に区別しているが、Skillの一部には「helperが明示trialsを上書き」「リークFAILEDは自動flatにならない」といった現在のコードと異なる説明が残る。正本を実装・テストに紐付け、現行状態の索引を更新すると、同じ修正を繰り返し提案するコストを減らせる。

**実施順序と完了条件。** 工数の大小だけでなく、次の意思決定を可能にするかで順序をつける。forwardデータ取得は期間を後から短縮できないため、基盤修正と並行して開始・継続する。

| 順序 | 実施する作業 | 得られる成果 | 完了条件 |
|---|---|---|---|
| 1 | 秘密情報の保存除外、全階層config検査、共通metrics | 誤記録・誤設定・誤比較の除去 | 合成再現が適切に拒否/除外され、既存正しい結果の意味が保たれる |
| 2 | 実時刻付きquoteとsource種別、snapshot固定 | 何をいつ知ったかの再現 | cutoff後観測を検出、proxy/始値代替を区別、全17銘柄の欠損を記録 |
| 3 | 約定・在庫・cash/fee照合と実口座loss stop | 実利益と実損失の把握 | unknown残高/費用が明示され、replayと異なる実損失でも停止可能 |
| 4 | forward on/offの日末結合、研究registry訂正、typed回帰 | 実験の再現と継続比較 | 同一日付・入力・費用仕様でreport/registry/日次台帳が一致 |
| 5 | 執行遅延・板・貸株・容量、carry/β診断 | 回収可能なαとリスクの評価 | 実単元・部分約定・費用条件下の制約とPnLが説明可能 |
| 6 | MH/ML/Ωの寄与診断と少数の新仮説 | 追加複雑性に見合う性能改善 | 未使用forward、事前基準、±感度・試行補正、DD/費用/fallbackを通過 |

最初の改修セットとしては、active fileの `experiment_utils.py` を中心に、秘密情報を除いた設定snapshot、MetricsSpecによる集計、NaNの扱い、study単位の記録を揃えるのが具体的である。設定loaderのstrict化は別の小さな差分に分ける。その後にquote時刻と台帳・risk接続へ進める。修正依頼時には対象回帰、全tests、compileall、既存CIのlint/型/import検査を行う。

**今は優先しない案。** JPX33/79/98分類への再拡大、LSTM/Transformer等への置換、特徴量の大量追加、同じ既知期間でのパラメータ総当たり、一律turnover制限の再試験、単純校正の再探索、停止を回避するためのVaR/ES閾値緩和、非同期基盤への全面移行は、現在の証拠では優先しない。現行17業種や感応度priorを最適と認定しているのではなく、変更の増分価値を検証できる条件を先に整えるという判断である。

**今回の検証と限界。** [review_probes.py](review_probes.py) を既存 `.venv`、60秒のプロセス全体期限、5秒の終了猶予で実行し、終了コード0を確認した。[probe_results.json](probe_results.json) に設定解決、誤字/存在しないpath、ダミー認証情報保存、未追跡code hash、NaN指標、初日DD、年率係数差を保存した。ダミーregistryとgit repositoryは `/tmp` の一時領域だけで使い、削除済みである。実API接続、実認証値の表示、実験registryへの追加、既存入力の書換えは行っていない。

再現コマンド：

```sh
MPLCONFIGDIR=/tmp/leadlag-analysis-mpl PYTHONDONTWRITEBYTECODE=1 timeout -k 5s 60s .venv/bin/python reports/20260928_project_improvement_analysis/review_probes.py
```

最新の本番状態・約定・scheduler結果を新たに照会したわけではない。引用したSharpe・VaR等は各レポートの期間・artifact・価格proxy・費用前提に限定され、将来の期待利益や現在稼働中の口座損益を保証しない。提案した新しいモデル・執行・リスク変更の性能効果は未検証である。

**2026-09-28 即時修正の追補。** 後続の修正依頼を受け、可逆で再現可能な不備を修正した。`src/research/experiment_utils.py` は `AppConfig` の全体dumpをやめ、研究用の許可リストsnapshotを保存する。dict設定は秘密情報らしいキーを再帰的に伏字にする。日次損益指標は共通 `MetricsSpec` / `calculate_metrics` に合わせ、年率係数とschemaを記録し、NaN/Infを含む系列は無効としてSharpe・DD・収益率を出さない。Stage5/6の集計・registryも各実験の252日年率仕様を明示して共通計算へ揃えた。過去のregistry記録は変更していない。

`src/leadlag/execution/config.py` はstrict時に既知の正規化対象セクションの未知項目・不正なmappingを拒否し、明示された存在しない設定パスを `FileNotFoundError` にする。strict検査は正規化対象セクションとその一部の下位mappingを対象としており、任意の未知YAML階層を全て再帰的に検査する実装ではない。`src/research/diagnostics/sprint1_experiments.py` は容量ゼロ時の未選択側除算を避け、診断テストで発生していたRuntimeWarningを除いた。変更済みVaR入力・履歴コードと同診断コードのimport整列も直した。

追補後の検証は、`pytest tests/ -n auto -q` で **924 passed**、Ruff（CI対象と変更した研究モジュール）、Mypy（`src/leadlag`）、import architecture contracts、compileallがすべて成功した。さらに設定・実験指標・Sprint1の関連テスト24件を `RuntimeWarning` をエラー扱いにして通過させた。全testsでは、意図的な定数入力による相関警告とテストfixture構築時のDataFrame fragmentation警告が残る。実バックテスト・発注・外部データ更新は行っていない。
