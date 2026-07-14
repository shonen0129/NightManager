# コード面改善 仕様書・メモ（B1–B7）

> 作成: 2026-07-13 / 上位モデルによる振り分け。
> 「仕様」= 本仕様に従い標準モデルで実装可能。「メモ」= 設計不要、標準モデルが単独で実行可能。

---

## B1. モンキーパッチ除去 — 【解決済みメモ】

**2026-07-13 検証結果: `sre.py` に `signals.build_c0_from_v0` のグローバル差し替えは既に存在しない。**
`compute_signal` の `c0_override` 引数（signal.py:39）経由に修正済みで、P4計算パスは
`compute_residual_signal(..., c0_override=C0_resid)`（sre.py:640-653）を使用している。

- **残作業（軽微）**: AGENTS.md「既知の落とし穴」のモンキーパッチ記述を削除・更新すること。
  なお `predict_signals` 内の `self.k = k_p4 / finally: self.k = orig_k`（sre.py:637-655）という
  **インスタンス状態の一時書き換え**は残っており、スレッド非安全性はここに残存。
  `compute_residual_signal` に `k_override` 引数を追加すれば除去できる（標準モデルで実装可）。

## B2. グローバルキャッシュ — 【メモ：標準モデルで実装可】

設計判断は単純: 6つのモジュールレベル dict を、モデルインスタンス保持の
`self._cache`（キーにデータ識別子 `hash(df_exec.index[0], index[-1], len)` を含める）へ移行し、
`predict_signals` 冒頭で clear。`_BASELINE_CORR_CACHE` は既にデータハッシュをキーに含むため優先度低。
特殊な理論判断は不要。refactor skill の手順に従い、テストを弱めずに実施すればよい。

## B3. 継承階層の composition 化 — 【設計仕様】

### 現状の問題

- `BaseModel` → `_BLPBase` → `BLPEnhanced` → `Bayesian` と、別系統 `BaseModel` → `SRE` が併存
- `predict_signals` が各層で全面オーバーライドされ、ループ本体が3箇所に重複
  （sre.py:617-677, blp_enhanced:1109-1265, bayesian:279-307）
- キャッシュ clear のタイミングが層ごとに不統一

### 目標アーキテクチャ

継承を「シグナル成分の合成」に置換する:

```
SignalComponent (Protocol)
  .compute(inputs: CommonInputs, i: int) -> np.ndarray
  実装: RawPCAComponent / ResidualPCAComponent / RawBLPXComponent /
        ResidualBLPXComponent / P4Component

EnsembleCombiner (Protocol)
  .combine(components: dict[str, np.ndarray], i: int) -> np.ndarray
  実装: StaticWeightCombiner / MetaLearningCombiner

SignalPipeline
  .__init__(components, combiner, config)
  .predict_signals(df_exec) -> dict   # ループは1箇所のみ
  内部: CommonInputs = prepare_common_inputs(df_exec, config)  # 純関数化
```

### 移行手順（後方互換を維持）

1. `prepare_common_inputs` を `blp_base.py` から純関数として `core/` に抽出
   （self 依存パラメータは dataclass `CommonInputsConfig` で受ける）。既存メソッドは薄いラッパーに
2. 各成分計算を `SignalComponent` 実装として抽出。既存クラスは Pipeline に委譲する
   ファサードとして温存（`predict_signals` の返り値 dict 形式を変えない — テスト互換）
3. 全テスト green を確認後、旧ループ本体を削除
4. B2 のキャッシュ移行は Pipeline のインスタンス属性として同時に実施すると重複作業を防げる

### 制約

- 返り値 dict のキー（`signals`, `normalized_signals`, `residual_blpx_signals` 等）は不変
- `ComplianceAuditor` / `v2_auditor` の監査フックポイントを移動しない
- 1 PR = 1 ステップ。各ステップで `bash scripts/run_tests_parallel.sh` 必須

## B4. 9:10価格近似 — 【メモ：データ依存、標準モデルで実施可】

理論判断は不要。手順: (1) 立花証券の実約定ログ（`live/`）と (H+L)/2 近似の乖離を
bps 単位で集計（B8 の R1 に対応）、(2) 乖離の符号・規模をレポート化、
(3) 系統的バイアスが片道 1bps を超える場合のみ、バックテストのコストパラメータ
（slippage_bps）に上乗せする**コスト側修正**を推奨。5分足 VWAP 等への執行モデル変更は
バックテスト再現性を壊すため、まずコスト側修正で十分。

## B5. 金利コスト日割り — 【メモ：標準モデルで実装可】

仕様は自明: `backtester.py` の `annual/365` 営業日課金を、ポジション保有の
**暦日数（次営業日までの実日数、金曜→月曜は3日）**で課金する方式に変更。
`market_calendar.py` が既にあるので日数計算はそれを利用。
変更前後の net Sharpe 差分を必ず報告（オーバーナイト保有が小さい現行構成では影響は
限定的なはずで、差分が大きい場合はバグを疑うこと）。

## B6. VaR99 の安定化 — 【設計仕様】

### 問題の定量化

250日窓の99%分位は順序統計量 X_(2)〜X_(3) 付近の補間で、標準誤差は分位点密度に反比例。
実効的に尾部2.5標本での推定 → stop 判定が単一外れ値の窓出入りで反転する。

### 推奨設計: 2段構え

1. **第1選択: Cornish-Fisher 展開**（実装コスト小・パラメトリック）

```
VaR_CF = −(μ + σ·z_cf),
z_cf = z + (z²−1)s/6 + (z³−3z)k/24 − (2z³−5z)s²/36
（z = Φ^{-1}(0.01)、s = 歪度、k = 過剰尖度、いずれも250日窓・外れ値3σ winsorize後）
```

   歪度・尖度は全標本から推定されるため尾部2.5標本問題を回避。
   有効領域チェック（s, k が CF の単調性領域外なら historical にフォールバック）を必ず入れる。

2. **第2選択: EVT-POT（GPD）** — CF が診断で不適合の場合のみ。
   閾値 = 90%分位（尾部25標本）、GPD を PWM 推定（MLE より小標本で安定）。
   実装・検証コストが大きいので CF で十分ならやらない。

### 検証プロトコル

- Kupiec POF 検定（violation 率 1% の尤度比検定）と Christoffersen 独立性検定を
  historical / CF の両方に適用し、バックテスト全期間で比較
- **stop 判定の反転頻度**（日次 VaR 推定の隣接日符号変化率）を安定性指標として併記
- ES は現行 tail-mean を維持（CF-ES は别式が必要なため第2段で検討）

### 互換性

`compute_var_es` の返り値 `VarEsResult` に `method` フィールドを追加し、
config で `var_method: historical | cornish_fisher` を切替可能に。既定は当面 historical
（本番 stop 基準の変更はシャドー検証を経ること）。

## B7. config shallow copy — 【メモ：標準モデルで実装可】

方針は単純: (1) `grep` で `cfg.copy()` / `config.copy()` を全箇所洗い出し
`copy.deepcopy` に置換、(2) 恒久対策として `leadlag/config/` に
`load_config(path) -> frozen Pydantic model` を整備し dict の裸回しを段階的に廃止
（ProductionV2RunConfig で先例あり）。比較実験スクリプトのテンプレートに
deepcopy を明記（AGENTS.md 既知の落とし穴に記載済み）。
