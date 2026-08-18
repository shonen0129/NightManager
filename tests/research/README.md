# tests/research/ — 研究・実験用テスト

このディレクトリには、研究スプリント (`sprint0`–`sprint3b`)、BLPX 診断、
実験アルゴリズムの回帰テストを配置する。

## 運用ルール

- `tests/research/` 配下のテストは **必ずしも本番V2パスを網羅しない**。
  本番V2の不変条件は `tests/integration/test_production_residual_blpx.py` 等を
  参照すること。
- 新しい研究実験を追加する場合は、ここにテストを追加し、**本番パスを触らずに**
  検証すること（AGENTS.md の改善ワークフロー1を参照）。
- テストの実行時間が長い場合は `pytest` の `slow` や `integration` マーカーを
  活用すること。

## 既存テスト群

- `test_sprint0_diagnostics.py`: 初期診断スクリプトの回帰テスト
- `test_sprint0_qa.py`: 初期QAテスト
- `test_sprint1.py`: Sprint1 バックテスト・キャリブレーション
- `test_sprint3b.py`: Sprint3b 特徴量変換
- `test_blpx_cost_consistency.py`: BLPX コスト整合性
- `test_backtester_910.py`: 9:10 価格調整のバックテスト
