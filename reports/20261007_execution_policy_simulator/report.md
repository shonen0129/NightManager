# 研究用執行ポリシー・板リプレイ

## 評価仮説と対象

仮説: spreadが広い、または執行側5段板のnotionalが注文量を覆わない場合、残り時間に余裕があれば同側最良気配で待機し、期限が近づいたら板内で成行性の高い注文へ切り替える。片側の約定でnetが偏る場合は、反対側を優先し、偏りを増やす側の次回許容量を縮める。

baselineは`broker_ops.build_execution_plan()`が作る現行の成行注文計画。シミュレータは同関数と`execution/microstructure/`の板スキーマ・spread・depth計算を使う。broker clientの生成や注文・取消・再送は行わない。

## リプレイデータの適格性

- 入力板ログ: `reports/20260923_profitability_order_2/quote_snapshots.jsonl`
- SHA-256: `cf0f6cea6fdf12277e94e3c09c3ef03b0ddba8dae6933213920e86bbf0d0aee7`
- 保存snapshot: 4件、完全な5段板: 4件
- 観測時刻: 2026-09-23T02:52:53.264117+09:00, 2026-09-23T02:53:46.999920+09:00, 2026-09-23T02:54:06.133346+09:00, 2026-09-23T02:58:45.073455+09:00
- 9:10適格snapshot: 0件、リプレイ適格イベント: 0件
- 受動約定のqueue証拠を含むsnapshot: 0件
- 実行状態: **NOT_EVALUABLE_NO_ELIGIBLE_REPLAY_EVENTS**

## baselineとの差

**差は算出できない。** 実行状態は `NOT_EVALUABLE_NO_ELIGIBLE_REPLAY_EVENTS`。入力の不足・不適格、または未確定の約定残量があるため、完全なshortfall差は報告しない。

行別の理由は`data_inventory.json`。比較が実行できた場合の既知部分は`baseline_vs_policy.json`に保存するが、未確定量を含む合計値とは区別する。

## 欠損と限界

- 到着基準板は09:10:00–09:10:30 JSTに限定する。同一セッションの後続板は期限・markout用に保持する。板5段より深い残量はunresolvedとして残す。
- quoteのtouchだけでは受動指値の約定とみなさない。各待機区間の完全な約定証拠が欠けたペアは、その後のクロスを停止し、後から完全なイベントが来てもunresolvedを解消しない。
- net exposureは既知約定の数量を各イベントmidで評価し、両方式とも約定順のピークと期限までの絶対値積分を計算する。midは次の観測または期限まで据え置く近似で、イベント内の実時間順序は観測していない。未確定量があればnet_exposure_statusもINCOMPLETE_DATAとする。
- execution shortfallは9:10 mid基準。約定分はfill時のdelayとspread/book costへ分解し、既知の未約定分は正確な期限midがある場合だけ機会費用化する。adverse selectionは別診断として報告し二重加算しない。
- broker fee、税、金利、貸株料、逆日歩はこの価格shortfallに含めない。該当する約定明細・費用データもこのリプレイ入力に存在しない。
- baseline注文は`broker_ops`のtarget/current差分から作る。比較には対象日と一致するsigned quantityの建玉snapshotが必要で、欠ければ空建玉を仮定せず停止する。
- `wide_spread_bps`, `urgent_last_seconds`, `max_pair_net_ratio`は初期設計値で、探索・感度分析・OOS検証をしていない。データ不足のため差分検定、Deflated Sharpe、ExperimentRegistryへの性能試行登録は行っていない。
- 保存ログの詳細な行別判定は`data_inventory.json`に記録した。
- 再現コマンド: `PYTHONPATH=src .venv/bin/python src/research/scripts/experiments/replay_execution_policy.py`
