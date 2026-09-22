# N1/N2 修正記録

実施日: 2026-09-21

## 修正内容

- ML overlay学習の`_collect_training_data()`を、日付とcacheパスだけの互換呼出から、BLPXをcanonical factoryで構築した`ProductionV2Model`と`DecisionInputs`を使う経路へ移行した。run所有の`open_910_returns`、macro、ADR、rank-reversal、PIT履歴と`HistoricalInputs.calculation_frame(as_of)`を渡すため、本番設定のh=1/3/5分布ブレンドと同じcache identity・PIT境界を通る。追加した補助入力hashもartifact metadataへ記録する。
- `HistoricalInputs.validate_observed_at(as_of)`を追加し、日付別または共通の補助入力観測時刻を`KnownMarketInputs.as_of`と照合する。`DecisionInputs`生成時に未来のmacro/ADR/rank/PIT/open_910等を拒否する。
- N2の未来観測拒否テストをS2bへ追加し、N1/N2の合成再現probeを修正後の期待値で回帰利用できるようにした。

## 検証

- 対象テスト: **45 passed**（S2b/S7）。
- 全体テスト: **845 passed / 17 warnings**。
- Ruff: 修正対象範囲PASS。
- 合成probe: 学習scoreと本番scoreの最大差 **0.0**、side不一致 **0件**。09:20観測のrank入力は`ValueError`で拒否。
- clean wheel検証（152 files、stale build output隔離・manifest・隔離smoke）もPASS。

## 残る確認

実データのverified artifact再生成・OOS評価・本番昇格、実scheduler/口座照合、Hosted CIは今回のローカル修正範囲外であり、S0–S8の外部受入条件として残る。
