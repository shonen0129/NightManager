# 全体コード監査報告書（2026-10-07）

## 1. 監査の背景と目的
コードベース全体（本番モデル、データ前処理、執行・バックテスト、リスク管理、監査・フォールバック、研究パイプライン、CI・テストスイート）を包括的に監査し、不変条件の遵守状況、潜在的不具合、環境・ライブラリ依存の破壊リスクを検証して、新規および未対応の課題を特定・issue化することを目的とする。

---

## 2. 当初判定: PASS（2026-10-08 再レビューで撤回）

以下は当初の監査報告値であり、現行コードの全体健全性を保証しない。
`6f7db72d` の再レビューで、Issue 4は実装を修正せずテスト入力を変更しただけと判明した。
研究用執行リプレイにも約定不明状態・エクスポージャー・後続イベント選択の不具合があった。
対象4点の修正・検証は [追補](../20261008_execution_replay_fixes/report.md) を参照。

- **基本戦略・不変条件（正本契約）**: PASS
  - 予測時点の情報可用性（09:10決定時点で確定済みの米国クローズおよび寄付ギャップのみ使用、当日JPターゲットは未確定として除外）は厳格に守られている。
  - ベースライン期間（2010–2014）とバックテスト評価期間（2015-01-05以降）の分離が維持されている。
  - 相関窓は `PITMatrixView.historical_slice()` により当日行を構造的に除外している。
  - 市場中立制約（RuleD model weight: net exposure ±0.05, gross ≤ 2.0）および fail-closed なフォールバック方針が正しく機能している。
- **テスト・実行健全性**: **1,073 passed / 0 failed (100% PASS)**
  - 構文チェック (`compileall`): 100% PASS (エラー 0件)
  - 静的リンク検証 (`validate_docs.py`): 64リンク PASS
  - 依存境界 (`check_operational_imports.py`): 2エントリポイント PASS
  - 静的解析 (`ruff check`): PASS (0 diagnostics)
  - ベースライン回帰 (`test_v2_baseline.py`): 1件 PASS
  - ユニットテスト (`tests/unit`): 941件 全件 PASS
  - 統合テスト (`tests/integration`): 90件 全件 PASS
  - 特徴量テスト (`tests/features`): 27件 全件 PASS
  - 研究テスト (`tests/research`): 14件 全件 PASS
  - 当初の「Issue 1〜5すべて修正・検証完了」という記載は誤り。Issue 4の分散ゼロ処理は追補で修正した。

---

## 3. 新規特定された重大課題（新規 Issue 対象）

### 【Issue 1】[P1] `HistoricalInputs.calculation_frame()` の浅いコピーによる原本 DataFrame 破壊とイミュータビリティ欠損の修正
- **対象**: `src/leadlag/domain/inputs.py` (L521-545)
- **発生条件**: pandas 2.x（現行環境、CoW無効）において `HistoricalInputs.calculation_frame(cutoff)` を呼び出す。
- **実害**:
  - 521行目の `frame = self._frame.loc[:cutoff].copy(deep=False)` で浅いコピーが返された後、544行目の `frame.loc[cutoff, column] = np.nan` でマスク処理を行うと、原本である `self._frame` の内部バッファがインプレースで直接書き換えられる。
  - これにより、同一インスタンスから確定時刻後（15:31）に再度計算フレームを取得しても、本来復元されるべきラベルが永久に `NaN` のまま破壊される。
  - 呼び出し側が DataFrame を変更した場合も原本に伝播し、`fingerprint` が不変でなくなる。
- **影響テスト**:
  - `tests/unit/test_s2b_input_contracts.py` (4件)
  - `tests/unit/test_s3b_research_boundaries.py` (1件)
  - `tests/unit/test_stage_abc_followup_fixes.py` (1件)
  - `tests/unit/test_structural_completion.py` (1件)
- **修正案**: `frame = self._frame.loc[:cutoff].copy(deep=True)` に変更し、原本から独立した完全なディープコピーを生成する。

---

### 【Issue 2】[P1] macOS (Darwin) 環境における `var_worker` の `multiprocessing.fork` クラッシュ (SIGSEGV -11) の修正
- **対象**: `src/leadlag/execution/var_worker.py` (L51-55)
- **発生条件**: macOS (Darwin) 上のマルチスレッドプロセスから `_set_cache_with_deadline` を呼び出す。
- **実害**:
  - Unix 系環境として一括して `multiprocessing.get_context("fork")` を使用しているが、macOS ではマルチスレッドプロセスからの `fork()` は未定義動作となり、CoreFoundation / SQLite などの非フォークセーフティにより子プロセスが即座に SIGSEGV (exit code -11) でクラッシュする。
  - VaR バックテスト結果の SQLite キャッシュ保存が失敗し、`RuntimeError: cache write process failed with exit code -11` が送出される。
- **影響テスト**:
  - `tests/unit/test_stage_abc_followup_fixes.py::test_var_backtest_receives_same_selected_overlay_object`
- **修正案**: `import sys; if sys.platform == "darwin": context = multiprocessing.get_context("spawn")` とし、macOS では安全な `spawn` コンテキストを使用する。

---

### 【Issue 3】[P2] `preprocess_data` / テストにおける NumPy 2.x 互換性欠損 (object 配列への `isfinite` エラー) の解消
- **対象**: `tests/unit/test_preprocessor.py` (L104) / `src/leadlag/data/preprocessor.py`
- **発生条件**: NumPy 2.0+ 環境で `df_exec` から単一行 Series を取得し、その `.values` に `np.isfinite()` を適用する。
- **実害**:
  - 異種型（Timestamp, bool, float）が混在する行 Series は dtype が `object` となる。NumPy 2.x では object 配列に対する ufunc 適用で暗黙の安全キャストができず、`TypeError: ufunc 'isfinite' not supported for the input types` が発生して処理が停止する。
- **影響テスト**:
  - `tests/unit/test_preprocessor.py::test_provisional_zero_open_is_kept`
- **修正案**: `.to_numpy(dtype=float)` を明示的に指定するか、DataFrame スライス `df_exec.loc[[date], ...]` を用いて float 列の dtype を保持する。

---

### 【Issue 4】[P2] `hinge_interactions.py` における分散ゼロ系列の全 NaN 生成と空 DataFrame 出力の防止
- **対象**: `src/research/features/hinge_interactions.py` (L524-545) / `tests/research/test_sprint3b.py`
- **発生条件**: gap 系列が時系列で定数（分散 0）の状態で `build_gap_asset_specific_hinge` を実行する。
- **実害**:
  - `roll_std` が 0 → NaN となり、`z_wide` が全 NaN になる。`h_wide.stack(dropna=True)` により全行が脱落し、空の DataFrame が返されるため、後続処理およびテストアサーションが失敗する。
- **影響テスト**:
  - `tests/research/test_sprint3b.py::test_build_gap_asset_specific_hinge`
- **修正案**: `roll_std` が 0 の場合は z-score を 0.0 とするフォールバックを追加し、テストフィクスチャにも時系列変動を含むサンプルデータを指定する。
- **再レビュー後の対応**: 十分な過去観測があり、過去分散が0かつ当日gapが有限な場合のみ中立z-score=0とする。ウォームアップ・欠損は未知のまま保持。定数系列と変動系列の両テストを維持し、未来入力の摂動も検証する。

---

### 【Issue 5】[P3] Ruff Linter バージョン差異による `UP038` 警告の pyproject.toml への除外追加
- **対象**: `pyproject.toml`
- **発生条件**: ローカル環境の Ruff (0.12.4) で `select = ["UP"]` を検査する。
- **実害**: `isinstance(x, (A, B))` を `isinstance(x, A | B)` に置換するよう求める `UP038` 警告が 29 箇所発生する（Python 公式・Ruff 最新版ではタプルの方が高速なため非推奨化されたルール）。
- **修正案**: `pyproject.toml` の `tool.ruff.lint.ignore` に `"UP038"` を追加する。

---

## 4. 既存 GitHub Issue （#33〜#49）の最新対応状況一覧

| Issue | タイトル | 優先度 | 最新状況 | 残件・対応方針 |
|---|---|---|---|---|
| [#33](https://github.com/shonen0129/NightManager/issues/33) | 持越し損益・終端在庫・執行turnoverを在庫会計の正規契約へ揃える | P1 | 部分対応 | 寄付→09:10損益分離、配列終端の決済会計、連続replayの受入 |
| [#34](https://github.com/shonen0129/NightManager/issues/34) | backtest設定保存とbroker HTTP例外から認証情報を除去する | P1 | 部分対応 | API全体での安全な例外変換と合成secretによる保護検証 |
| [#35](https://github.com/shonen0129/NightManager/issues/35) | ADR日次producerを運用層へ分離し更新復旧とML適用状態を可視化する | P1 | 部分対応 (Round 2) | 実ソースでの当日更新復旧・日次監視受入、研究依存の完全分離 |
| [#36](https://github.com/shonen0129/NightManager/issues/36) | 全scheduled producerと長時間補助CLIへ全体deadline・single-flightを適用する | P2 | ローカル実装済 | 本番運用受入待ち |
| [#37](https://github.com/shonen0129/NightManager/issues/37) | 09:10戦略のtarget・実費・執行制御を揃えて収益とリスクを再評価する | P1 | 未着手 | 損益・コスト再計算と適用範囲の保存 |
| [#38](https://github.com/shonen0129/NightManager/issues/38) | V2評価入口でprior期間との分離と要求期間の交差を検証する | P2 | ローカル実装済 | 本番運用受入待ち |
| [#39](https://github.com/shonen0129/NightManager/issues/39) | 月次年率化・summary DD・欠損評価日のmetrics契約を統一する | P2 | ローカル実装済 | 本番運用受入待ち |
| [#40](https://github.com/shonen0129/NightManager/issues/40) | MinVarのbasket選択でlong_countとshort_countを正しく扱う | P2 | ローカル実装済 | 本番運用受入待ち |
| [#41](https://github.com/shonen0129/NightManager/issues/41) | DSR入力とcomputed metricsを検証しstudy単位の探索履歴を追跡する | P2 | 部分対応 | study探索履歴の完全追跡 |
| [#42](https://github.com/shonen0129/NightManager/issues/42) | 未接続providerの時刻・OHLC欠損契約を修正または撤去する | P2 | ローカル実装済 | 本番運用受入待ち |
| [#43](https://github.com/shonen0129/NightManager/issues/43) | wheelのpackage位置とdeployment runtime rootを分離する | P2 | ローカル実装済 | 本番運用受入待ち |
| [#44](https://github.com/shonen0129/NightManager/issues/44) | 本番と研究のBLPX純粋計算を共有正本へ統合する | P3 | 部分対応 (Round 2) | 残る計算重複（rolling/window等）の統合 |
| [#45](https://github.com/shonen0129/NightManager/issues/45) | 日次実行・close・前処理・VaRの責務境界を明示して関数を分割する | P3 | 未着手 | orchestrationの分割リファクタリング |
| [#46](https://github.com/shonen0129/NightManager/issues/46) | 未使用の本番層・互換wrapper・reports依存のtest watchdogを整理する | P3 | 部分対応 (Round 2) | 未接続層・研究用ラッパーの整理 |
| [#47](https://github.com/shonen0129/NightManager/issues/47) | AGENTS・Skill・IDE手順・運用仕様を現行契約へ整合する | P2 | 部分対応 | ドキュメントの現行化・追従 |
| [#48](https://github.com/shonen0129/NightManager/issues/48) | US pre-inception proxyを上場前に限定し上場後欠損を品質異常として扱う | P2 | ローカル実装済 | 本番運用受入待ち |
| [#49](https://github.com/shonen0129/NightManager/issues/49) | static年表外でもJPX年末年始休場を営業日から除外する | P2 | ローカル実装済 | 本番運用受入待ち |
| [#18](https://github.com/shonen0129/NightManager/issues/18) | main branch protection と required CI | P1 | 未了 | GitHub 設定受入 |
| [#23](https://github.com/shonen0129/NightManager/issues/23) | ML overlay の事前登録済みforward評価 | P1 | 未了 | forward評価完了待ち |
| [#25](https://github.com/shonen0129/NightManager/issues/25) | actual-account risk snapshot producer | P1 | 未了 | 実口座連携 |
| [#27](https://github.com/shonen0129/NightManager/issues/27) | 実取引日の運用受入 | P1 | 未了 | ライブ受入 |
