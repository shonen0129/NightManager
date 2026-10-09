# ML overlayを研究専用に戻し、本番判断をV2単独にする（2026-10-09）

## 決定

本番の意思決定は `ProductionV2Model` 単独にする。`configs/production/production.yaml`
で `ml_overlay_enabled: false` を明示し、overlay artifact rootを本番設定から外した。
本番model factoryは無効設定ではartifactをロードしない。

ML overlayの学習・artifact・推論機能は研究再現用に保持する。candidate artifactは
`configs/research/ml_overlay_forward_shadow_20261009.yaml` からだけ選択し、実際の
本番V2結果と同一 `DecisionInputs` でpaired shadowを計算する。shadow計算は同じV2設定・費用・
riskからoverlay設定だけを変え、candidate予測・失敗は本番ウェイト、ポートフォリオ出力、発注へ
反映しない。shadow失敗は記録・ログに分離し、production decisionを継続する。

`scripts/batch/run_decision_v2.sh` は従来どおり09:10の読み取り専用quote/板captureを行い、
research shadow configを渡す。新しい追記系列は
`var/shadow_runs/ml_overlay_research_20261009/` に分離し、従来のpaired記録を混ぜない。
quote/板snapshotは従来どおり `var/shadow_runs/ml_overlay_value/microstructure/` から参照する。
`--shadow-only` は本番V2 baselineと研究candidateだけを計算し、
本番portfolio出力・建玉照会・発注へ進まない。

## 理由

事前登録したforward採用gateには、train-end後250日の完全なpaired forward評価、net PnL差の
block-bootstrap 95%区間下限>0、net Sharpe差>0、最大DD悪化1pp以内、turnover増加10%以内、
実約定費用との照合、監査PASSが必要である。現行artifactでは前向き標本とprovider
`available_at`来歴の条件が未達で、250日VaR/ES履歴も停止基準を超過している。2026-09-27の
operator-directed promotionは数値gate合格ではない。これらを根拠にcandidateを本番経路から外し、
V2単独を本番基準に戻す。履歴上のpromotion記録とimmutable artifactは監査・研究用に保持する。

## 研究評価の契約

- live baselineはその日の本番V2結果、candidateはresearch configで計算したoverlay結果。
- 両者は同一の入力digest、frozen quote snapshot、trade dateに結び付ける。
- config fingerprintには両設定とcandidate artifact metadataを記録する。
- overlay計算・保存が失敗した場合は不完全なshadow rowを残し、本番出力や発注判断を変更しない。
- shadowのモデル費用は実約定費用ではない。採否判断は既存のforward gateと実費照合が満たされるまで保留する。
- 本番へ戻す操作はこの決定では行わない。採用時は別途OOS結果・監査・リスク確認と明示的な設定変更を必要とする。

## 影響範囲と確認

本番切替は設定のみで行い、BLPX/V2・RuleD・risk stop・side leverageは変更しない。
candidate artifactと過去のpaired shadow/outcome記録は削除・改変しない。
継承解決後のproduction configがoverlay無効であり、model factoryがcandidate artifactを
ロードしないことを回帰テストで確認する。paired shadowテストではproduction V2結果をbaselineとして
保持し、research candidateの失敗を本番側へ伝播させない。
