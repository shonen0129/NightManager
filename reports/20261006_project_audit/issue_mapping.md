# 全体監査のGitHub issue対応表（2026-10-06）

親tracker: [#22](https://github.com/shonen0129/NightManager/issues/22)。新規17件、既存4件へ追記。監査31項目をすべて対応付けた。

| 監査ID | 追跡先 | 優先度 |
|---|---|---|
| F01, F18, F19 | [#33: [P1] 持越し損益・終端在庫・執行turnoverを在庫会計の正規契約へ揃える](https://github.com/shonen0129/NightManager/issues/33) | P1 |
| F02 | [#34: [P1] backtest設定保存とbroker HTTP例外から認証情報を除去する](https://github.com/shonen0129/NightManager/issues/34) | P1 |
| F05, F22 | [#35: [P1] ADR日次producerを運用層へ分離し更新復旧とML適用状態を可視化する](https://github.com/shonen0129/NightManager/issues/35) | P1 |
| F06 | [#36: [P2] 全scheduled producerと長時間補助CLIへ全体deadline・single-flightを適用する](https://github.com/shonen0129/NightManager/issues/36) | P2 |
| F07, F09, F10 | [#37: [P1] 09:10戦略のtarget・実費・執行制御を揃えて収益とリスクを再評価する](https://github.com/shonen0129/NightManager/issues/37) | P1 |
| F11, F12 | [#38: [P2] V2評価入口でprior期間との分離と要求期間の交差を検証する](https://github.com/shonen0129/NightManager/issues/38) | P2 |
| F13, F14, F17 | [#39: [P2] 月次年率化・summary DD・欠損評価日のmetrics契約を統一する](https://github.com/shonen0129/NightManager/issues/39) | P2 |
| F15 | [#40: [P2] MinVarのbasket選択でlong_countとshort_countを正しく扱う](https://github.com/shonen0129/NightManager/issues/40) | P2 |
| F16, F28 | [#41: [P2] DSR入力とcomputed metricsを検証しstudy単位の探索履歴を追跡する](https://github.com/shonen0129/NightManager/issues/41) | P2 |
| F20 | [#42: [P2] 未接続providerの時刻・OHLC欠損契約を修正または撤去する](https://github.com/shonen0129/NightManager/issues/42) | P2 |
| F21 | [#43: [P2] wheelのpackage位置とdeployment runtime rootを分離する](https://github.com/shonen0129/NightManager/issues/43) | P2 |
| F23 | [#44: [P3] 本番と研究のBLPX純粋計算を共有正本へ統合する](https://github.com/shonen0129/NightManager/issues/44) | P3 |
| F24 | [#45: [P3] 日次実行・close・前処理・VaRの責務境界を明示して関数を分割する](https://github.com/shonen0129/NightManager/issues/45) | P3 |
| F25 | [#46: [P3] 未使用の本番層・互換wrapper・reports依存のtest watchdogを整理する](https://github.com/shonen0129/NightManager/issues/46) | P3 |
| F26, F27 | [#47: [P2] AGENTS・Skill・IDE手順・運用仕様を現行契約へ整合する](https://github.com/shonen0129/NightManager/issues/47) | P2 |
| F30 | [#48: [P2] US pre-inception proxyを上場前に限定し上場後欠損を品質異常として扱う](https://github.com/shonen0129/NightManager/issues/48) | P2 |
| F31 | [#49: [P2] static年表外でもJPX年末年始休場を営業日から除外する](https://github.com/shonen0129/NightManager/issues/49) | P2 |
| F03 | [#27（既存へ追記）](https://github.com/shonen0129/NightManager/issues/27#issuecomment-6007169543) | P1 |
| F04 | [#25（既存へ追記）](https://github.com/shonen0129/NightManager/issues/25#issuecomment-6007171006) | P1 |
| F08 | [#23（既存へ追記）](https://github.com/shonen0129/NightManager/issues/23#issuecomment-6007172092) | P1 |
| F29 | [#18（既存へ追記）](https://github.com/shonen0129/NightManager/issues/18#issuecomment-6007173778) | P1 |

各issueに根拠・再現・影響・完了条件を記載。旧issueの元の完了範囲は維持。依存関係と優先順は親trackerに記載。


## 2026-10-06 ローカル修正の到達点

[修正結果・issue別の残件](../20261006_issue_resolution/report.md) と [検証概要](../20261006_issue_resolution/verification.json) を追加した。
#36/#38/#39/#40/#42/#43/#48/#49 の主不具合をローカルで修正。#33/#34/#35/#41/#46/#47 は部分対応として追跡する。GitHub issueのclose・本番受入は実施していない。
