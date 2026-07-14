# B3: 継承階層のComposition化 — 設計提案書

> 作成: 2026-07-14 / 上位モデルレビュー済み
>
> **判定: 条件付き承認。現案をそのまま実装してはならない。**
> 以下の「上位モデルレビュー決定」を正本とし、旧案の疑似コードは概念説明としてのみ扱う。

## 1. 現状分析

### 1.1 継承階層

```
BaseModel (abstract: predict_signals)
├── SectorRelativeEnsembleModel (sre.py)
│   └── 独自 _prepare_common_inputs (US残差化P4対応)
└── _BLPBase (shared: _prepare_common_inputs, compute_production/residual_signal,
    │          _denormalize_signal, _apply_gap_adjustment)
    ├── SectorRelativeEnsembleBLPModel (blp.py)
    ├── SectorRelativeEnsembleBLPEnhancedModel (blp_enhanced.py)
    │   └── BayesianBLPXModel (bayesian_blpx.py)
    └── SectorRelativeEnsembleRRRModel (rrr.py)
```

`production_v2.py` に `ProductionV2Model` クラスは存在しない。同ファイルはgap調整後の
ポートフォリオ構築関数群であり、この継承階層の外側にある。本番バックテストは
`SectorRelativeEnsembleBLPEnhancedModel` を直接生成するため、B3の主な本番影響先は
同モデルと `BacktestEngine` / decision consumer である。

### 1.2 `predict_signals` ループ重複（5箇所）

| モデル | ループ行数 | コンポーネント | 状態更新 | 返り値dictキー |
|--------|-----------|--------------|---------|--------------|
| SRE | ~60行 | RawPCA, ResidualPCA, P4 | なし | raw_pca, residual_pca, p4, signals, normalized |
| BLP | ~70行 | RawPCA, ResidualPCA, P5, P5P3 | なし | raw_pca, residual_pca, p4(空), p5, p5p3, signals, normalized, blp_diagnostics |
| BLPEnhanced | ~180行 | RawPCA, ResidualPCA, RawBLPX, ResidualBLPX | meta-learning展開窓 | raw_pca, residual_pca, p4(空), raw_blpx, residual_blpx, signals, normalized, sigma_yy, blp_diagnostics |
| Bayesian | ~70行 | ResidualBLPXのみ | Kalman gain, IC history, B_bayes_prev, error_history | signals, normalized, residual_blpx, raw_pca(=residual), p4(=residual), raw_blpx(=residual), sigma_yy, blp_diagnostics, bayesian_diagnostics |
| RRR | ~100行 | RawPCA, ResidualPCA, P6, P6P3, P7, P7P3 | なし | raw_pca, residual_pca, p6, p6p3, p7, p7p3, signals, normalized, rrr_diagnostics |

### 1.3 `_prepare_common_inputs` の2系統問題

- `sre.py`: US残差化(P4)対応版 — `all_returns_p4`, `r_us_adj`, `spy_returns` を追加生成
- `blp_base.py`: 基本版 — P4データを生成しない
- RRRモデルは `blp_base.py` 版を `super()._prepare_common_inputs()` 経由で使用しつつキャッシュ

### 1.4 B2完了後の状態（2026-07-13）

全グローバルキャッシュはインスタンス属性化済み。407テスト全通過。
グローバル→インスタンス変換は不要だが、Component抽出時に**キャッシュ所有権とrun境界を
そのまま移管する作業は残る**。特に同一モデルインスタンスでの連続実行、別データ実行、
例外後の再実行でキャッシュとオンライン状態が混線しないことを保証する。

## 2. 提案アーキテクチャ

### 2.1 コア型定義

```python
# core/pipeline.py

from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass(frozen=True)
class CommonInputsConfig:
    """prepare_common_inputs のパラメータを純データとして表現"""
    n_u: int
    n_j: int
    ewma_half_life: int
    beta_window: int
    include_v4_prior: bool
    us_res_enabled: bool
    us_res_gamma: float
    us_res_beta_window: int
    prior_variant: str | None

@dataclass
class CommonInputs:
    """prepare_common_inputs の出力（純データ）"""
    all_returns_raw: np.ndarray
    c_full: np.ndarray
    c_full_p3: np.ndarray
    v0_static: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    jp_gap: np.ndarray
    jp_beta: np.ndarray | None
    topix_night: np.ndarray | None
    y_jp_oc_df: pd.DataFrame
    jp_res_returns_p3: np.ndarray
    y_jp_target: np.ndarray
    # Optional P4 fields
    all_returns_p4: np.ndarray | None = None
    r_us_adj: np.ndarray | None = None
    spy_returns: np.ndarray | None = None

def prepare_common_inputs(df_exec: pd.DataFrame, cfg: CommonInputsConfig) -> CommonInputs:
    """純関数: df_exec と config から CommonInputs を生成"""
    ...

@runtime_checkable
class SignalComponent(Protocol):
    """シグナル成分のインターフェース"""
    name: str  # "raw_pca", "residual_blpx", etc.
    def compute(self, inputs: CommonInputs, i: int) -> np.ndarray:
        """index i でのシグナルを返す（shape=(n_j,)）"""
        ...

@runtime_checkable
class StatefulSignalComponent(Protocol):
    """ループ内で状態更新を行う成分（Bayesian等）"""
    name: str
    def compute(self, inputs: CommonInputs, i: int) -> np.ndarray:
        ...
    def reset(self) -> None:
        """predict_signals 開始時に呼ばれる状態リセット"""
        ...

@runtime_checkable
class EnsembleCombiner(Protocol):
    """成分シグナルを統合するインターフェース"""
    def combine(self, components: dict[str, np.ndarray], i: int) -> np.ndarray:
        ...

class SignalPipeline:
    """単一ループで全成分を計算し統合する"""
    def __init__(
        self,
        components: list[SignalComponent],
        combiner: EnsembleCombiner,
        config: dict,
        corr_window: int,
        normalization_method: str = "zscore",
    ):
        self._components = components
        self._combiner = combiner
        self._corr_window = corr_window
        self._normalization_method = normalization_method

    def predict_signals(self, df_exec: pd.DataFrame, inputs: CommonInputs) -> dict:
        T = len(df_exec)
        sim_dates = df_exec.index
        start_idx = self._corr_window

        # 状態リセット
        for comp in self._components:
            if hasattr(comp, "reset"):
                comp.reset()

        # 出力配列初期化
        component_signals = {comp.name: np.zeros((T, inputs.all_returns_raw.shape[1] - inputs.n_u if hasattr(inputs, 'n_u') else 17)) for comp in self._components}
        combined = np.zeros((T, len(component_signals[list(component_signals.keys())[0]])))
        normalized = np.zeros_like(combined)

        for i in range(start_idx, T):
            sigs = {}
            for comp in self._components:
                sigs[comp.name] = comp.compute(inputs, i)
                component_signals[comp.name][i] = sigs[comp.name]

            s_ens = self._combiner.combine(sigs, i)
            combined[i] = s_ens
            normalized[i] = normalize_signals(s_ens, self._normalization_method)

        # 返り値dict構築（モデル固有のキー互換性はFacadeで処理）
        ...
```

### 2.2 Component実装例

```python
class RawPCAComponent:
    name = "raw_pca"
    def __init__(self, cfg: dict):
        self.k = cfg.get("k", 6)
        self.lambda_reg = cfg.get("lambda_reg", 0.75)
        # ... 全パラメータ
        self._cache: dict = {}  # インスタンスキャッシュ

    def compute(self, inputs: CommonInputs, i: int) -> np.ndarray:
        cache_key = (i, self.lambda_reg, self.k, ...)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        sig = signals.compute_signal(
            all_returns=inputs.all_returns_raw,
            current_index=i,
            k=self.k,
            c_full=inputs.c_full,
            v0_static=inputs.v0_static,
            v1=inputs.v1,
            v2=inputs.v2,
            ...
        )
        self._cache[cache_key] = sig
        return sig

class ResidualBLPXComponent:
    name = "residual_blpx"
    def __init__(self, cfg: dict):
        # BLPパラメータ
        ...
    def compute(self, inputs: CommonInputs, i: int) -> np.ndarray:
        # compute_blp_signal(is_residual=True) を呼ぶ
        ...

class BayesianResidualBLPXComponent:
    """BayesianBLPXModel の状態更新付き成分"""
    name = "residual_blpx"  # 同名でBLPEnhanced版を置換
    def __init__(self, cfg: dict):
        self.bayesian_enabled = cfg.get("bayesian_enabled", True)
        self.bayesian_mode = cfg.get("bayesian_mode", "ic")
        # ... 全Bayesianパラメータ
        self._B_bayes_prev = None
        self._ic_history = []
        self._eta_history = []
        self._prev_predicted_z = None
        self._cs_var_history = deque(maxlen=...)
        self._error_history = deque(maxlen=...)
        self._B_struct_history = deque(maxlen=...)

    def reset(self) -> None:
        self._B_bayes_prev = None
        self._ic_history = []
        self._eta_history = []
        self._prev_predicted_z = np.zeros(self.n_j)
        self._cs_var_history.clear()
        self._error_history.clear()
        self._B_struct_history.clear()

    def compute(self, inputs: CommonInputs, i: int) -> np.ndarray:
        # y_actual_prev = inputs.y_jp_target[i-1] if i > start_idx + 1 else None
        # → compute_blp_signal_bayesian と同等の処理
        ...
```

### 2.3 Combiner実装例

```python
class StaticWeightCombiner:
    """固定重みの線形結合"""
    def __init__(self, weights: dict[str, float]):
        self._weights = weights

    def combine(self, components: dict[str, np.ndarray], i: int) -> np.ndarray:
        result = np.zeros_like(next(iter(components.values())))
        for name, sig in components.items():
            result += self._weights.get(name, 0.0) * sig
        return result

class MetaLearningCombiner:
    """BLPEnhancedのmeta-learning combiner"""
    def __init__(self, cfg: dict):
        self.meta_enabled = cfg.get("meta_learning_enabled", False)
        self.meta_train_window = cfg.get("meta_learning_train_window", 252)
        self._meta_weights = None  # 展開窓

    def combine(self, components: dict[str, np.ndarray], i: int) -> np.ndarray:
        if not self.meta_enabled:
            # StaticWeightCombiner に委譲
            ...
        # IC計算 → w_t予測 → 動的重み付け
        ...
```

## 3. 移行手順（後方互換維持）

### Step 1: `prepare_common_inputs` 純関数化（低リスク）

- `core/pipeline.py` に `prepare_common_inputs(df_exec, cfg) -> CommonInputs` を作成
- `sre.py` と `blp_base.py` の `_prepare_common_inputs` は薄いラッパーに変更
- 既存のインスタンスキャッシュは `CommonInputsConfig` をキーに引き継ぎ
- **テスト互換**: メソッドシグネチャ不変、返り値dict互換

### Step 2: SignalComponent Protocol + RawPCA/ResidualPCA抽出

- `RawPCAComponent`, `ResidualPCAComponent` を実装
- `SignalPipeline` の基本ループを実装
- 既存モデルの `predict_signals` は Pipeline を使うように変更
- **テスト互換**: 返り値dictのキーは各モデルのFacadeで保証

### Step 3: BLP系Component抽出

- `RawBLPXComponent`, `ResidualBLPXComponent` を実装
- `BLPEnhanced` の `predict_signals` を Pipeline に移行
- meta-learning combiner を実装

### Step 4: Bayesian状態更新統合

- `BayesianResidualBLPXComponent` を実装（`StatefulSignalComponent`）
- Pipeline のループ開始時に `reset()` を呼ぶ
- `BayesianBLPXModel` の `predict_signals` を Pipeline に移行

### Step 5: RRR・旧ループ削除

- RRR Component を実装
- 全モデルの旧 `predict_signals` ループ本体を削除
- `_BLPBase` の `compute_production_signal` / `compute_residual_signal` は Component に委譲

## 4. 上位モデルに相談したい設計判断

### Q1: Bayesian状態更新のPipeline統合方式

**問題**: `BayesianBLPXModel.predict_signals` はループ内で以下の状態を更新する:
- `self._B_bayes_prev` — 前ステップのBayesian更新後B行列
- `self._ic_history` — Rolling Rank IC履歴
- `self._error_history` — 予測誤差履歴（Kalman gain計算用）
- `self._B_struct_history` — B_struct履歴（プロセスノイズQ推定用）
- `self._prev_predicted_z` — 前ステップの予測値（誤差計算用）

これらは `compute(i)` の中で `inputs.y_jp_target[i-1]` を使って更新される。
`SignalComponent.compute(inputs, i)` のインターフェースで `inputs.y_jp_target` にアクセス可能だが、状態更新をComponent内に閉じ込めてよいか？

**提案**: `StatefulSignalComponent` Protocol に `reset()` を追加し、Pipeline開始時に呼ぶ。状態はComponentインスタンスが保持。`compute()` 内で `inputs.y_jp_target[i-1]` を参照して更新。

**懸念**: テストが `BayesianBLPXModel` のインスタンス属性（`self._B_bayes_prev` 等）を直接参照している場合、Component内に移動するとアクセスパスが変わる。

### Q2: 返り値dictのキー互換性

**問題**: 5モデル全てで返り値dictのキーが異なる:

```python
# SRE
{"raw_pca_signals", "residual_pca_signals", "p4_signals", "signals", "normalized_signals", ...}

# BLPEnhanced
{"raw_pca_signals", "residual_pca_signals", "p4_signals", "raw_blpx_signals", "residual_blpx_signals", "signals", "normalized_signals", "sigma_yy", "blp_diagnostics"}

# Bayesian
{"signals", "normalized_signals", "residual_blpx_signals", "raw_pca_signals"(=residual), "p4_signals"(=residual), "raw_blpx_signals"(=residual), "sigma_yy", "blp_diagnostics", "bayesian_diagnostics"}
```

**提案**: Pipeline は統一的な中間フォーマット（`{component_name}_signals`）を返し、各モデルクラスのFacadeが旧キー形式に変換する。

**懸念**: Bayesianは `raw_pca_signals` に `residual_blpx_signals` と同じDataFrameを入れている（テスト互換のため）。このエイリアスをFacadeで再現する必要がある。

### Q3: meta-learningの展開窓配置

**問題**: BLPEnhancedの `predict_signals` ループ内で:
- `us_dispersions`, `cond_nums`, `vix_vals`, `ic_blpx_vals`, `ic_pca_vals` を日次で蓄積
- `i >= start_idx + meta_train_window` で `_predict_meta_weight` を呼んで `w_t` を予測
- `w_t` は `combine_signals` の重みを動的に変更

これを `MetaLearningCombiner.combine(components, i)` に押し込める場合、Combinerが状態（IC履歴等）を持つことになる。

**提案**: `EnsembleCombiner` も `StatefulSignalComponent` と同様に `reset()` を持たせる。Pipeline開始時に `combiner.reset()` を呼ぶ。

### Q4: `_prepare_common_inputs` の2系統統合

**問題**: `sre.py` 版はUS残差化データ（`all_returns_p4` 等）を追加生成するが、`blp_base.py` 版は生成しない。

**提案**: `CommonInputsConfig.us_res_enabled` が `True` の場合のみP4データを生成。`CommonInputs` のP4フィールドは `None` 許容。各Componentは必要に応じて `inputs.all_returns_p4` を参照（`None`ならスキップ）。

### Q5: ステップ分割とPR戦略

**提案**: 5ステップを別々のPR/commitに分割。各ステップで `bash scripts/run_tests_parallel.sh` 必須。

**懸念**: Step 2とStep 3の間で、Pipelineと旧ループが共存する期間がある。この期間のテスト安定性をどう保証するか。

## 5. リスク評価

| リスク | 影響度 | 対策 |
|--------|-------|------|
| 返り値dictのキー不整合 | 高 | Facadeパターンで旧キーを再現、テストで検証 |
| Bayesian状態のアクセスパス変更 | 中 | Decoratorへ移し、旧メソッドは移行期間中委譲で維持 |
| meta-learning展開窓のバグ | 高 | Stateful Combiner化し、`i-1` 境界をgolden masterで固定 |
| パフォーマンス劣化 | 低 | Component呼び出しとキャッシュhit率を移行前後で測定 |
| 本番Residual-BLPX経路への影響 | 高 | `run_production_backtest.py` の出力完全一致をモデル移行ゲートにする |

## 6. 上位モデルレビュー決定（正本）

### 6.1 結論

B3は実装可能。ただし、現案の `compute(...) -> np.ndarray` と単一の汎用ループでは情報が不足する。
BLPX/RRR/Bayesianはシグナル以外に共分散、条件数、行列、フォールバック情報、オンライン更新診断を
返しており、それらを失う設計は不可。以下の契約に変更する。

### 6.2 正式な実行契約

```python
@dataclass(frozen=True)
class RunContext:
    dates: pd.DatetimeIndex
    n_u: int
    n_j: int
    start_idx: int
    start_idx_raw: int

@dataclass(frozen=True)
class StepContext:
    run: RunContext
    inputs: CommonInputs
    i: int

@dataclass
class ComponentResult:
    signal: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    covariance: np.ndarray | None = None

class SignalComponent(Protocol):
    name: str
    def begin_run(self, context: RunContext) -> None: ...
    def compute(self, context: StepContext) -> ComponentResult: ...
    def end_run(self) -> dict[str, Any]: ...

class EnsembleCombiner(Protocol):
    def begin_run(self, context: RunContext) -> None: ...
    def combine(
        self,
        context: StepContext,
        components: Mapping[str, ComponentResult],
    ) -> ComponentResult: ...
    def end_run(self) -> dict[str, Any]: ...
```

決定事項:

1. `ComponentResult` を必須とし、`np.ndarray` 単体を返す契約は禁止する。
2. `hasattr(reset)` のような暗黙判定は禁止する。全Component/Combinerが明示的な
   `begin_run` / `end_run` を実装し、stateless実装はno-opとする。
3. `start_idx` と `start_idx_raw` はPipelineが一律に決めない。モデルFacadeが現在の規則を
   正確に計算して `RunContext` に渡す。`_start_date` 最適化も保持する。
4. Pipelineは計算順を設定として受ける。Bayesianとmeta-learningでは順序自体が状態遷移の一部であり、
   自動並列化・遅延評価・成分順のソートは禁止する。
5. 正規化はPipelineにハードコードせず、Combinerまたは明示的なNormalizerに委譲する。

### 6.3 Q1回答: BayesianはDecorator Componentにする

`BayesianResidualBLPXComponent` がBLPX計算を再実装する案は却下する。
`BLPXComponent` を内包するDecoratorとして実装する。

```python
BayesianUpdateComponent(base=ResidualBLPXComponent(...), updater=BayesianUpdater(...))
```

`BayesianUpdater` の状態はUpdaterインスタンスが保持し、`begin_run` で必ず初期化する。
`compute` は当日行を学習に使わず、現行どおり `y_jp_target[i - 1]` と前回予測だけで更新する。
境界条件も現行の `i > start_idx + 1` をgolden masterで固定する。

既存テスト・本番コードにBayesian内部属性の直接参照は見つからないため、Facadeへのproperty転送は
原則不要。ただし公開互換を優先し、移行期間中は旧メソッド
`compute_blp_signal_bayesian` をDecoratorへの薄い委譲として残す。

### 6.4 Q2回答: 出力互換は専用OutputAdapterで固定する

Pipelineがモデル固有キーを知る案と、Facade内にアドホックなdict変換を散在させる案は避ける。
モデルごとに `OutputAdapter` を1つ置く。

- `SREOutputAdapter`
- `BLPOutputAdapter`
- `BLPXOutputAdapter`
- `BayesianBLPXOutputAdapter`
- `RRROutputAdapter`

Adapterはキーだけでなく、型、shape、index、列順、ゼロ埋め、Bayesianの既存エイリアスも再現する。
`BacktestEngine` が要求する `raw_pca_signals`, `residual_pca_signals`, `p4_signals`, `signals`,
`normalized_signals`, `y_jp_oc_df` は全Facadeで契約テストを追加する。

### 6.5 Q3回答: meta-learningはStateful Combinerでよい

`MetaLearningCombiner` がIC履歴、VIX、condition number、前回weightを保持する設計を承認する。
ただし `components: dict[str, np.ndarray]` ではcondition numberを受け取れないため、
`Mapping[str, ComponentResult]` を受け取る。`y_jp_target[i - 1]` の参照は `StepContext` 経由とし、
当日ターゲット参照を禁止する。

VIX/macroデータのロードはCombiner内で行わない。run開始前にInputBuilderが取得済み系列を
`CommonInputs` に格納する。これにより計算ループ内I/Oと隠れたbfillを分離できる。

### 6.6 Q4回答: 完全な純関数化ではなくInputBuilderを採用する

現行 `compute_jp_target_returns` は内部で5分足キャッシュをロードするため、単に関数へ移しても
純関数ではない。以下の2層に分割する。

1. `CommonInputBuilder`: データ取得・5分足ロード・任意macro系列ロードを担当するrun-scoped object。
2. `build_common_inputs(df_exec, intraday_5m, cfg)`: 明示入力だけから `CommonInputs` を作る計算関数。

P4は「Optionalフィールドを自由に読む」のではなく、`P4Inputs` の明示的サブ構造にする。
P4 Component生成時に `P4Inputs` が無ければ初期化時点でfail-fastし、実行中に黙ってスキップしない。
これは本番の「gap欠損時はフラット」とは別層の設定不整合である。

`CommonInputs` には `n_u`, `n_j`, `dates` を明示保持する。配列shapeから17を推測するコードや
ハードコードfallbackは禁止する。

### 6.7 Q5回答: 7段階に分割する

1. **Characterization tests**: 変更前の全モデル出力を固定。キー、型、shape、index、値、診断、
   repeated-run、別インスタンス分離をテストする。
2. **InputBuilder抽出**: 既存 `_prepare_common_inputs` はdictを返す薄いAdapterとして残す。
3. **PCA Component抽出**: まだ既存ループから直接呼ぶ。Pipelineへ切り替えない。
4. **SREのみPipeline化**: 最小モデルでPipelineとOutputAdapterを検証する。
5. **BLP/RRR移行**: stateless系を先に移す。モデルごとに独立PRとする。
6. **BLPX移行**: static combiner → macro処理 → meta combinerの順で分ける。
7. **Bayesian移行と継承除去**: Decorator移行後、最後の別PRで旧継承・旧ループを削除する。

Step 2と3の共存は意図的なstrangler移行とする。新経路はfeature flagで切り替えず、
移行対象モデル単位でFacadeの委譲先を一度だけ変更する。旧経路とのA/B比較テストをPR内に置き、
一致確認後に次のモデルへ進む。

### 6.8 必須の振る舞い一致ゲート

通常の全テストgreenだけでは不十分。各移行PRで以下を追加する。

- 同じ `df_exec` に対する旧/新の全出力配列を `rtol=0`, `atol=1e-12` で比較。
- DataFrameのindex・columns・dtype・NaN位置を比較。
- diagnosticsのキー集合と数値を比較。
- 同一インスタンスで2回実行した結果が一致すること。
- データA実行後にデータBを実行しても、fresh instanceのB結果と一致すること。
- `len(df_exec) < corr_window`、`len == corr_window`、最初の有効日を検証。
- 各rolling sliceが `[window_start:i]` で当日行を除外すること。
- Bayesian/metaで行 `i` の重み・更新に使えるtargetは最大でも `i-1` であること。
- baseline 2010–2014が存在し、先頭1260行fallbackが発動しないこと。
- `ComplianceAuditor` / `v2_auditor` の既存フックと監査結果を維持すること。
- 全テスト `bash scripts/run_tests_parallel.sh` を通すこと。

バックテストのnet Sharpe比較だけでは微小なシグナル差を見逃すため、B3ではまず完全な配列一致を
合格条件とする。意図的な浮動小数点順序変更が必要な場合のみ、理由を記録して許容誤差を緩和する。

### 6.9 実装開始判定

- **今すぐ着手可**: Step 1（characterization tests）とStep 2（InputBuilder抽出）。
- **Step 3以降の前提**: Step 1のgolden masterが全モデル分揃っていること。
- **禁止**: 最初のPRでProtocol、Pipeline、全Component、全Facadeを同時導入すること。
- **禁止**: B3とモデルパラメータ、リーク境界、コスト計算、シグナル数式を同時変更すること。

## 7. 現状のコード規模

- `sre.py`: 699行
- `blp_base.py`: 238行
- `blp.py`: 310行
- `blp_enhanced.py`: 1309行
- `bayesian_blpx.py`: 352行
- `rrr.py`: 663行
- **合計**: ~3,571行（うちループ重複約500行）

Composition化で推定~500行削減、新規Pipeline/Component定義で~300行追加、純削減~200行。
