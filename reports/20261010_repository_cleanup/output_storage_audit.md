# 出力ディレクトリ圧縮の監査（2026-10-10）

## 容量と扱い

- `reports/` は1,315 files、83.7 MiB（Git管理1,295 files）。検証・実験の証跡と文書参照を保つため、圧縮せず元のパスに残した。
- 圧縮前の `results/` は17,384 files、論理5,090,872,621 bytes（約4.74 GiB）。そのうち今回の7ディレクトリは390 files、4,490,416,228 bytes。
- 7ディレクトリを、元の相対パスを保持する単一tar.gzへ置き換えた。圧縮後は906,373,990 bytes（約0.84 GiB、原本論理サイズの20.18%）。

## 対象範囲

置き換えたのは次のGit管理外の履歴結果ディレクトリだけ。

- `results/blp_enhanced_refined/`
- `results/baseline_reconciliation/`
- `results/etf_basket_substitution/`
- `results/sector_relative_ensemble_pls_extended/`
- `results/sre_etf_vs_stock_basket/`
- `results/rrr_projection/`
- `results/production_shadow_run_package/`

現行コードから読む日次 `*_production_*` スナップショット、旧実験が参照する `results/gap_adjusted_distribution/`、研究設定の出力先 `results/us_residual_prior/` / `results/us_residualization/`、その他未調査の `results/` は保持した。

## 検証と復元

- iCloudから対象全体を取得し、390ファイルのサイズとSHA-256を記録した。
- アーカイブ内の全ファイル名・サイズ・SHA-256とディレクトリ構成を照合し、390/390件が一致した。最終保存先のアーカイブもSHA-256を再確認した。
- すべての検証が通った後に限り、対象7ディレクトリを削除した。対象外の保持ディレクトリが残っていることも確認した。
- macOS tarが自動生成するAppleDoubleメタデータ項目は除外した。照合対象は元ファイルのバイト内容と相対パス。
- アーカイブ: [`var/results/historical_research_results_20261010.tar.gz`](../../var/results/historical_research_results_20261010.tar.gz)
- SHA-256マニフェスト: [`historical_results_manifest.json`](historical_results_manifest.json)
- アーカイブSHA-256: `03caae916dbdac55889648a5df0c7051944a98104d70eab5dbb74795b9f54ea5`
- iCloud Driveにはアーカイブ項目が登録された。最後に応答した転送状態は32.78%（311,994,120 / 951,853,668 bytes）で、その後のFile Provider状態照会が応答せず、クラウド側の全量同期完了は未確認。ローカルの検証済みアーカイブは保持されている。
- リポジトリルートでの復元コマンド: `tar -xzf var/results/historical_research_results_20261010.tar.gz -C .`

初回のiCloud取得では読み取り待ちとファイル更新時刻の変化があり、またmacOS tarの追加メタデータを厳密な照合が検出した。いずれも原本保持のまま処理を停止し、原因に合わせて再実行した。最終検証が成功する前に原本を削除したことはない。
