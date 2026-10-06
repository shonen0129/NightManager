# ADR公開とBLPX計算の正本（2026-10-06）

Status: accepted for local implementation; live recovery acceptance pending.

## 背景

監査issue #35/#44/#46では、ADR運用updaterのresearch import、pickle/CSVの個別直書き、欠損のゼロ化、BLPX計算の複製、旧PnL/gap wrapperと未使用CostCalculatorを指摘した。[監査報告](../../reports/20261006_project_audit/report.md)の元の証拠は保持する。

## 判断

- ADR生成は `leadlag.data.adr_producer` に置く。計算はclose表とUS/JP日付対応を明示入力で受け取り、ネットワーク取得と公開を分離する。銘柄対応は `data/tickers.py` の `ADR_SECTOR_MAP` / `ADR_TICKERS` が正本。ETFの15/17次元は変えない。
- US signal日はJP trade日より前でなければならない。未来価格・未知のJP当日targetを使わず、US closeのpct_changeは補間しない。未対応セクターはcoverage=0の構造的ゼロ、対応セクターは観測できた有限returnの平均と観測数を保持する。対応ADRが全欠損ならNaNを保持する。
- 当該取引日の全17特徴が有限で、signal provenanceとcoverageが一致する場合だけ公開する。historical欠損は保存し、live/backtestのその日とADR付き学習ではskipする。少なくとも1 ADRの観測を対応セクターごとに要求し、完全な全銘柄取得を仮定しない。Yahooの日次indexだけでは9:10以前の提供可用性を証明できない。
- 正規artifactはruntime rootの `data/adr_features.zip`。pickle/CSVとschema version 1のmanifestをまとめ、同じdirectoryのtemporary fileをfsyncして一度のos.replaceでcommitする。manifestは公開時刻、最終trade/signal日、行数、historical不完全行数、当日セクター別coverage、両payloadのSHA-256を持つ。readerはhashと内容・manifestの整合を確認し、破損時はNoneを返す。
- 旧pickle/CSVを読む互換fallbackや、zero-imputedな旧artifactの自動移行は設けない。再取得の正常公開まではMLがskipし、次の正常公開で回復する。実data取得・scheduler起動を今回のオフライン受入と混同しない。
- scheduled updaterは従来のjob guard/phase deadlineを維持する。network requestには既存timeout定数を使い、daemon wait timeout後に遅延threadが公開することはない。ADR単独復旧入口も同じleaseと全体deadlineの下で実行する。
- 7つのBLPX純粋計算を `core/blpx_math.py` に移し、モデルinstanceを受け取らず係数と次元を明示入力で渡す。旧helper wrapper/re-exportは撤去する。研究の診断キー `z_U` は本番の `z_U_t` へ統一し、旧名は残さない。
- solve/pseudo-inverseが非有限の結果を返す場合もfallbackを扱い、ゼロ行列とfallback=trueを返す。特異行列以外の通常計算の演算順は変えない。既存の数値監査・リーク監査は保持する。
- PnL callerは `core.pnl.simulate_daily_pnl`、gap callerは4値の `load_gap_bundle` を使う。旧3値wrapperで既定だったmetadata必須を移行先の各該当testで明示し、provenance gateを維持する。使用元がtestのみのCostCalculatorを撤去し、使用されるbps関数の費用・暦日・LOB/fallback回帰を直接検証する。

## 検証と残件

変更前のcommitを独立moduleとしてロードし、独立config/model/inputで数値比較する。詳細と再現コマンドは [追加対応報告](../../reports/20261006_issue_resolution_round2/report.md)。#35の実更新復旧、#44のwindow/prior構成・signal orchestration等の残る重複、#46のprovider/convex optimizer/ML wrapperは別途追跡し、Phase全体の完了は宣言しない。
