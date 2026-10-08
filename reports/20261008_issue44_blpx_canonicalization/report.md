# Issue #44: BLPX 計算の共通化

## 対応内容

- `leadlag.core.blpx_math` を本番・研究BLPXの数値計算の正本にした。共通化した処理は係数solve、固定sector priorの構築、rolling sector priorのblend、PCA/Tikhonov、confidence weighting、非対称solve、signal/gap変換、診断構築。
- 共通のUS-to-JP sector mappingを `leadlag.data.tickers` に集約した。production/research modelの `_SECTOR_MAPPING_STRUCTURE` 属性は残し、実験ごとの上書きを可能にしている。
- 本番・研究のsignal entry pointは、それぞれのwindow preparation、correlation estimation、sector-prior hook、非対称共分散推定を呼んだ後、明示入力で共通計算へ渡す。
- `combine_signals` とpipeline callbacksはモデル別の合成責務として保持した。`_get_sector_prior` もmodel override可能なhookとして残し、デフォルトのprior計算だけを共有化した。
- 旧数値計算の重複実装と `_build_sector_prior` は削除した。数値計算を複製する互換wrapperは追加していない。

## 変更前後の数値比較

変更前の基準は `f08d1fd9117eb12594f7fde4ee78f795d9ee1000`。このcommitを独立した一時checkoutで実行し、保存済みfixtureを作成した。固定seedの640×32 returns、signal index 560、504行の過去window、固定gap/betaとPCA prior入力を使い、production/research両方の結果を比較する。

比較ケースは有限入力とPSD相関、window内のNaN/Inf、対象windowより前の値だけを変えた入力、covariance asymmetry modeの4種。signal、標準化forecast、診断値と `return_matrices=True` が返す全16行列・ベクトルを照合し、各値の許容差をabsolute `1e-12`、relative `0` とした。finite caseの相関行列最小固有値も変更前値と照合する。さらに、future rowsを書き換えても結果が変わらないこと、windowより前のprefix変更で結果が変わらないこと、非有限window入力から有限な行列・signalが返ること、両モデルのprior hookが実際に使われることを回帰テストに含めた。

fixtureは `tests/fixtures/issue44_blpx_signal_baseline.json`、比較とhook回帰は `tests/research/test_blpx_shared_signal.py`。

## 検証

- 対象BLPX回帰: 24 passed。
- 固定fixture比較: 4ケース×2モデルについてsignal、診断値、返却された全16行列・ベクトルをabsolute `1e-12` / relative `0` で照合。未来入力とwindow外prefixに対する不変性、非有限入力の有限出力、prior hookも確認。
- 全テスト（回帰ベースラインを除く）: 1,145 passed、既存警告2件。
- `tests/regression/test_v2_baseline.py`: 1 passed。
- `compileall`: 成功。
- production/researchのRuff checksと `git diff --check`: 成功。
- post-change AST clone scanではprior blendとsignal orchestrationの数値実装重複は残らない。検出された同形関数はモデルhook adapter、signal entry point、モデル固有のensemble composition/pipeline callbacksで、共通計算へ委譲する役割または各モデル固有の構成を持つ。
- mypyは実行環境に `mypy` moduleと `uv` がなく起動できなかったため未検証。
