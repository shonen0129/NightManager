# サブセクター精緻化予測レイヤー 設計・実験計画

**Date**: 2026-08-22
**Status**: 計画（未実装）
**参照**: `configs/taxonomy_subsectors.yaml`（79 サブセクター / 498 銘柄）、`docs/モデル技術仕様書.md`、`docs/ARCHITECTURE.md`

---

## 1. 背景・目的

現行モデル（Production Residual-BLPX-RA v2）は **米国 15 セクター ETF のクローズ・トゥ・クローズリターン** から **日本 TOPIX-17 セクター ETF の翌営業日 9:10→大引けリターン** を直接予測している。予測空間は 17 次元であり、1 つのセクター内に US 感応度の異なる企業群が混在する（例: 1626 情報通信・サービスには NTT のようなディフェンシブ通信キャリアと任天堂・ソニーのようなグローバルコンテンツが同居する）ため、BLP の投影がセクター内平均に潰され、米国側の細かい産業シグナルを取りこぼしている可能性がある。

**提案**: `configs/taxonomy_subsectors.yaml` の細分類（79 サブセクター / 498 個別銘柄の時価総額加重バスケット）を予測空間に使い、**サブセクター単位で値動きを予測した後、各銘柄が属する TOPIX-17 セクターへ時価総額加重で集約して 17 次元の予測に戻す**。取引は従来通り TOPIX-17 セクター ETF（1617.T–1633.T）のみ行う。

### 1.1 仮説

- **H1（主仮説）**: セクター内の US 感応度異質性が大きいセクター（情報通信・サービス、電機・精密、素材・化学、小売等）では、サブセクター経由の予測が直接 17 セクター予測より情報量を持ち、集約後の 17 次元予測 IC が改善する。
- **H0（帰無仮説）**: セクター内の US 感応度異質性は小さく、サブセクター化は推定次元を 17→79 に増やす分だけ推定誤差を増やし、改善はノイズマージン内（または悪化）に収まる。**この結論も価値がある**（不採用なら `docs/experiment_graveyard.md` に記録し再検証を防ぐ）。

なお、サブセクター化で期待できるのは **米国感応度の異質性の取り込み** であり、個別株ノイズの分散低減ではない（バスケット化・セクター集約で個別株の idiosyncratic 成分は既に平均化されるため、そこに追加の改善は理論上ほぼない）。この点を明確にしておくことで、結果解釈のブレを防ぐ。

### 1.2 スコープ

| 項目 | 扱い |
|:---|:---|
| 予測空間 | サブセクター時価総額加重バスケット（最大 79 次元。推定・集約とも cap-weighted、等加重は感度比較） |
| 取引ユニバース | **変更なし**（TOPIX-17 ETF 17 銘柄、net ±0.05 / gross ≤ 2.0 制約も不変） |
| 米国シグナル源 | **変更なし**（US 15 ETF のみ。SMH/IGV 等の米国側細分化は本実験では行わない — §10 将来拡張） |
| gap 調整・執行 | **変更なし**（17 セクター ETF の 9:10 価格・`jp_gap_*` を使用。サブセクター個別の 9:10 価格は本実験の第1版では使わない） |
| 本番パス (`src/leadlag/`) | 実験期間中は変更しない（`src/research/` に閉じ込める） |

---

## 2. 現行パイプラインと差分

### 2.1 現行（baseline）

```
US 15 ETF cc ──┐
               ├─ EWMA 相関 C (32×32, 窓504, 半減期120)
JP 17 oc ──────┘   （JP 側は 9:10→大引け oc ターゲット（9:10 不在時 open→close）、
                    TOPIX 残差化済み、β窓60・1日ラグ）
      │
      ▼ ブロック縮約 (αxx=0.20, αyx=0.15, αyy=0.50) + Ridge (ρ=0.01)
B_blp = Σ_YX A⁻¹            （17×15）
      │
      ▼ 構造的縮小: B_struct = (Σ_YX + λ_pca B_pca + λ_sector M_sector) A_tikh⁻¹
ẑ_J = B_struct z_U          （17 次元）
      │
      ▼ 確信度調整 (β_conf=0.25) → 逆標準化 (σ²⁰) → 翌朝 gap 調整 (c=0.70, b=0.6)
信号 s (17) → gap 調整済み予測分布 (mu_gap, Omega_gap)
      → mu_over_sigma ランキング → L/S 各5銘柄 → MinVar (α=0.8) → RuleD グロス → w_final
```

該当コード: `src/leadlag/models/blpx/signal_computer.py::compute_blp_signal`、`src/leadlag/models/blpx/prior_builder.py`（`_build_sector_prior` / `_SECTOR_MAPPING_STRUCTURE`）、`src/leadlag/core/correlation.py`（`compute_baseline_correlation` / 事前部分空間）、`src/leadlag/models/v2/`（分布解決・フォールバック）。

### 2.2 提案（差分は ★）

```
US 15 ETF cc ──┬──────────────────────────────┐
               │                              │
               ├─ [既存] 17 セクター BLPX ─────┤
               │        → ẑ_direct (17)       │
               │                              │
               └─ ★ サブセクター BLPX          │
                    SUB 79 oc（open→close、    │
                    TOPIX 残差化）※推定ターゲット│
                    は本番に合わせ cc でなく oc │
                    EWMA 相関 (15+79=94 次元)  │
                    → B_struct_sub (79×15)     │
                    → ẑ_sub (79)               │
                                                 │
               ★ 集約: ẑ_agg = A ẑ_sub  (A: 17×79 カバレッジ重み行列)
               ★ 合成: ẑ_s = (1−w_s)·ẑ_direct,s + w_s·ẑ_agg,s
                    （w_s = セクター別カバレッジ連動。非カバーセクターは w_s=0 で direct のみ）
      │
      ▼（ここから下流は一切変更なし）
確信度調整 → 逆標準化 → gap 調整 → v2 分布 → ポートフォリオ構築
```

**設計上の要点**: サブセクター層は「17 次元 ẑ_J を置き換える」のではなく「17 次元 ẑ_J を精緻化して返す」関数として挿入する。これにより v2 分布 (`mu_gap`, `Omega_gap`)、mu_over_sigma、MinVar、RuleD、監査・フォールバックといった下流機構を一切変更せず、不変条件 4（市場中立制約）とフォールバック挙動を構造的に保全する。

---

## 3. サブセクター ↔ TOPIX-17 マッピング（非全射の扱い）

### 3.1 用語と前提

- 各 **個別銘柄** はちょうど 1 つのサブセクターとちょうど 1 つの TOPIX-17 セクターに属する（TOPIX-33 業種分類から 17 へのロールアップで一意に決まる）。
- **サブセクター → 17 セクター** の対応は一般に多対多になりうる（例: 「Global Gaming Platforms」は任天堂=その他製品→1626 と、ソニー=電気機器→**1625** を跨ぐ）。
- **非全射**: `taxonomy_subsectors.yaml` は全市場を網羅しない（498 銘柄は TOPIX 構成銘柄の一部であり、カウントベースでは過半未満）。17 セクターの中にはサブセクター定義が全く無い、またはカバレッジが極端に小さいセクターがありうる（正確な非カバー / カバレッジ状況は Phase 0 の §3.5 検証で確定する）。

### 3.2 集約行列 A の構成（銘柄レベル定義でクロスマッピングを自然に処理）

各銘柄 i に対し (subsector k(i), TOPIX-17 セクター s(i), 集計重み a_i) を定義し、

```
A[s, k] = Σ_{i: s(i)=s, k(i)=k} a_i ／ Σ_{i: s(i)=s} a_i      （分母が 0 の行 s は「非カバー」）
```

- **集計重み a_i = 時価総額（PIT 近似）** とする（2026-08-22 指定）。ETF の cap-weighted ターゲットと整合し、カバレッジ `cov_s` の時価総額ベース測定と一体となる。PIT 時価総額の構築方法と既知の近似限界は §4.4 を参照。等加重は感度分析の比較案として残す
- 行ごとに正規化されるため、A の各行は「カバー銘柄が属するサブセクター予測の凸結合」になる。クロスマッピング銘柄は自身の真の TOPIX-17 セクターの行にのみ寄与する
- **具体例**（サブセクター A=(aaa1, aaa2)、B=(bbb1, bbb2)、ETF1=(aaa1, bbb1)、ETF2=(aaa2, bbb2) の場合）:
  ```
  pred(ETF1) = [cap(aaa1)·ẑ_A + cap(bbb1)·ẑ_B] / [cap(aaa1) + cap(bbb1)]
  pred(ETF2) = [cap(aaa2)·ẑ_A + cap(bbb2)·ẑ_B] / [cap(aaa2) + cap(bbb2)]
  ```
  すなわち「予測はサブセクター単位で行い、ETF へのマッピングは各 ETF の構成銘柄が属するサブセクター予測の時価総額加重平均」。分母はカバー銘柄のみ（未カバー部分は `w_s` で direct へ割り戻し、§3.3）
- **カバレッジ指標**: セクター s のカバレッジ `cov_s` = **カバー銘柄の時価総額合計 / セクター全体の時価総額**（主指標。セクター全体の時価総額は JPX 業種別構成データまたはセクター ETF 構成銘柄から構築）。カウントベース（銘柄数比）は参考指標として併記する

### 3.3 非全射・部分カバレッジの扱い

| ケース | 扱い |
|:---|:---|
| セクター s が非カバー（A の行が空） | `w_s = 0`（直接 17 セクター予測のみ使用）。フォールバックはセクター単位で独立に動作 |
| 部分カバー（cov_s < 1） | `w_s = w_max × clip(cov_s, 0, 1)` とし、カバレッジが低いほど direct を重視。`cov_s` は §4.4 の PIT 時価総額パネルから時変計測（月次更新で churn 抑制）を主とし、静的スナップショットは感度比較。`w_max` は既定 0.5（新規スカラーパラメータ 1 個、±20% 感度分析対象） |
| サブセクター k が当日無効（構成銘柄のデータ不足） | ẑ_sub,k を NaN とし、A の該当列を除外して行を再正規化（残りのサブセクターで凸結合）。有効列が 0 なら当該セクター w_s=0 |

### 3.4 設計判断: 全射の確保 vs 全 TOPIX 拡張（2026-08-22 検討結果）

**検討された代替案**: 「取引対象（TOPIX-17 ETF）は TOPIX 全体を母集団とするのに、taxonomy は ≈TOPIX500（498 銘柄）に留まるのだから、先にサブセクター分類を TOPIX 全体へ拡張して全射・全カバーにすべきではないか」。

**結論: 全射は確保するが、全 TOPIX 拡張は第1版では行わない。** 理由:

1. **ターゲットは時価総額加重**: TOPIX-17 ETF は cap-weighted。TOPIX500 と TOPIX のリターンはほぼ一致（JPX ファクトシート: 1年 17.63% vs 17.53%、10年 309% vs 305%）し、498 銘柄で TOPIX 全体の時価総額の大半（推定 9 割台半ば、Phase 0 でセクター別に実測）を既にカバーしている。未カバー ~1,100 銘柄の限界寄与は小さい
2. **小型株は米国感応度が低い**: 本戦略の情報源は米国セクターリターンであり、国内・固有要因支配の小型株はサブセクター予測のα寄与が小さくノイズが勝つ
3. **サバイバーシップ悪化**: 小型株ほど 2010–2026 期間の廃止・買収・破綻が多く、現時点分類の履歴投影バイアス（§4.3）と yfinance 欠損が増大する
4. **分類品質の空洞化**: 1,100+ 銘柄の手作業分類は誤分類リスクが大きく、JPX 33 業種コードへの機械割当は「経済的に意味のある細分化」という taxonomy 自体の価値を尾部で失わせる
5. **推定ノイズ**: サブセクター数 79→120+ で最小バスケットがさらに狭小化し、94 次元の推定誤差問題（§8-1）が悪化する

**代わりに確保すること**: (a) **全射性** — 17 セクター全てに ≥1 個の有効サブセクターが写ることを Phase 0 ゲートとし、未カバーセクターがあれば当該セクター限定で最小限のサブセクターを追加する（全拡張ではない）。(b) **カバレッジの計測と反映** — `cov_s` を時価総額ベース（主）とカウントベース（参考）で計測し、`w_s = w_max × cov_s` で未カバー分を direct 予測へ適切に割り戻す（§3.3）。

**全 TOPIX 拡張の発動条件（将来フェーズ）**: Phase 1–2 で本機構の有効性が確認され、かつ coverage 分析で「cov_s が低いせいで w_s が頭打ちになり P&L 上重要なセクターが direct 依存になっている」ことが具体的に示された場合に限り、**当該セクターに限定して** JPX 33 業種コードによる機械割当で拡張を検討する。

### 3.5 マッピングの正本構築（Phase 0 の成果物）

1. **正本ファイル**: `src/research/experiments/subsector/taxonomy_map.py` が `configs/taxonomy_subsectors.yaml` を読み、銘柄 → TOPIX-17 の対応表を生成する。TOPIX-17 業種の根拠は JPX 33 業種コード（取得元: JPX 上場会社一覧 / 業種コードマスタ）→ 17 セクターロールアップ表で固定し、**手動マッピングは原則禁止**（クロスマッピングの取り違え防止）。
2. **生成物**: `configs/research/subsector_mapping_generated.yaml`（銘柄 × {subsector, topix17, 33業種コード}）+ 検証レポート（サブセクター × 17 セクターのクロス集計表、非カバーセクター一覧、カバレッジ一覧（カウント / 時価総額の両ベース））。
3. **検証**: 時価総額加重バスケットの過去リターンと対応する TOPIX-17 ETF リターンの相関を算出し、全セクターで十分に正であること（目安 ρ ≥ 0.7。未満のセクターは要調査）を Phase 0 の品質ゲートとする。

---

## 4. データ要件と構築

### 4.1 必要データ

| データ | 期間 | 用途 | 取得元 |
|:---|:---|:---|:---|
| 498 銘柄の日次 OHLC（調整済み終値 + 生 OHLC。**Open 必須**） | 2010-01-01 以降 | サブセクター推定ターゲット（open→close oc リターン）・TOPIX 残差化・ベースライン相関 (2010–2014)・参考用 cc | yfinance（既存 `YFinanceProvider` 系を流用。タイムアウト対策必須） |
| TOPIX 指数（1306.T） | 同上 | サブセクターの TOPIX 残差化 β 推定 | 既存データで賄える |
| JPX 業種コードマスタ | 最新スナップショット | §3.5 マッピング構築 | JPX 公表情報（手動取得） |

- **9:10 価格（5分足）は不要**（第1版）。ただし **推定ターゲットは cc ではなく open→close（oc）** とする: 本番パイプラインは JP 側の結合行列に `jp_oc_*`（9:10→大引け、9:10 不在時は open→close フォールバック）を使っている（`blp_base.py::_prepare_common_inputs` → `pipeline.py::_build_all_returns_raw`）ため、サブセクター側も oc で推定しないと direct モデルと推定空間がずれ、§5.2 のブレンドの整合が崩れる。個別株の 9:10 mid は歴史分がないため、サブセクター oc は日次 OHLC の Open で近似する（本番のフォールバック定義と同じ）。
- **定義差のロバスト性確認（Phase 0/1 で実施）**: ETF 側の直近期 `jp_oc_*` は真の 9:10 mid（5分足 (High+Low)/2）を使う期間があり、open ベースとの定義差が残る。ETF の oc を open ベースで再計算して本番系列と比較し、推定相関への影響が小さいことを確認する。
- バックテスト開始日不変（2015-01-05）。ベースライン期間 2010–2014 の分離を維持するため、504 営業日の初期推定窓を考慮しても 2010 年からのデータで充足する。

### 4.2 サブセクターバスケット時系列の構築ルール

- サブセクター k の日次リターン: `r_k,t = Σ_{i∈k, available_i,t} cap_i,t · r_i,t / Σ_{i∈k, available_i,t} cap_i,t`（**時価総額加重**、§4.4 の PIT 時価総額 + **アベイラビリティマスク付き**。cap 欠損銘柄は除外）。**oc（open→close）と cc の 2 系列を構築する** — 推定に使うのは oc（§4.1 の本番整合理由）、cc は参考診断用。等加重版は感度比較用に別系列として保持
- 時価総額加重のトレードオフ: 3–4 銘柄の小バスケットでは最大銘柄に支配され「バスケット」が実質単一銘柄化しうる。§4.2 の最少有効銘柄数ルールと合わせ、集中度（最大重み > 70% 等）をログに記録して診断する
- **アベイラビリティマスク**: 銘柄 i は上場日以降かつデータ有効日のみ available。新規上場（例: キオクシア 285A=2024 年、東京地下鉄 9024=2024 年、JX 金属 5016、ブルーゾーン 417A 等）は上場前 NaN。`preprocess_data` の規約に倣い、サブセクター内の有効銘柄が **半数未満または 2 銘柄未満** の日は当該サブセクターを NaN（無効）とする
- **trade-date 整合**: パネルは `df_exec` の `jp_oc_*` と同じ意味論（行 t = 取引日 D_{t+1} の oc リターン）で trade-date index にアラインする。US cc（シグナル日 D_t）とのリードラグ対応は `df_exec` の行構造をそのまま踏襲し、独自の日付ずらしを行わない
- 保存形式: `var/research/subsector/panel_subsector_oc.parquet`（推定用）+ `panel_subsector_cc.parquet`（参考）+ `mask_subsector.parquet`（bool 行列）+ 構築ログ（欠損率・除外銘柄一覧）。**`live/pipeline_data/` には置かない**（運用データ保護規約）

### 4.3 サバイバーシップ / メンバーシップの既知の限界（要正直な記述）

`taxonomy_subsectors.yaml` は **現時点のメンバー表** であり、2015 年時点のセクター構成を厳密には再現しない（除退場銘柄が含まれず、新規上場銘柄は上場後のみ有効）。これは軽度のサバイバーシップ / ルックアヘッド（現時点の分類知識の使用）を伴う。

- **影響の性質**: 対象は大型流動性銘柄中心でセクター平均からの乖離は限定的と予想されるが、定量化は必須。
- **緩和策**: (1) Phase 1 で「直近 5 年（2021–2026）のみのサブ検証」を行い、メンバーシップ誤差が小さい期間で効果が保つか確認。(2) バックテストの IC/Sharpe 改善が直近年に偏っていないか年次分解で確認。(3) レポートに本制約を明記し、改善幅の割引要因として扱う。

### 4.4 PIT 時価総額の構築（バスケット重み・集計重みの原料）

バスケット構成（§4.2）と ETF 集約（§3.2）の両方の重みに使う時価総額系列を構築する。

- **定義**: `cap_i,t = Close_i,t（生価格）× Shares_i,t`（日次）。Adj Close（配当調整込み）は使わない（時価総額に配当調整は不要）
- **株式数の PIT 近似**: yfinance の現在の発行済株式数を基準に、株式分割履歴で過去へ遡及調整する（`Shares_i,t ≈ Shares_now × Π_{τ>t} split_factor_τ`）。増資・自社株買いによる株式数ドリフトは捕捉できない既知の近似誤差（大型株では年数 % 程度。重みはセクター内相対値なので影響は二次的）
- **浮動株調整は不可**: TOPIX 本体は浮動株ベースだが、浮動株比率の歴史は無償データでは再現困難。上場株式数ベースの近似であることを明記
- **ルックアヘッド性**: 現在の株式数を過去へ投影するため、§4.3 と同種の軽度のルックアヘッドを含む。感度分析で「PIT 近似時価総額 / 固定スナップショット時価総額 / 等加重」の 3 案を比較し、重み付け選択が結果を左右しないことを確認する
- 有効性ルール: `cap_i,t` が欠損・非正の銘柄は当日のバスケット・集約から除外（アベイラビリティマスクと連動）

---

## 5. モデル設計（数理）

### 5.1 サブセクター BLPX

現行 Residual-BLPX の定式化を次元 94（US15 + SUB79）に適用する。機械は `ProductionBLPXModel` を再利用（`n_u`/`n_j` は cfg から変更可能: `src/leadlag/models/blpx/model.py` L91-92）。

1. **推定ターゲットの整合（重要）**: サブセクター側の Y は **open→close（oc）リターン**とする（§4.1 参照。本番の JP 側結合行列は cc ではなく oc ターゲット）。各サブセクターの oc リターンを TOPIX に対し残差化（ローリング OLS β、窓 60、1 日ラグ。現行 §2.1 と同一手順をサブセクターに適用）
2. **EWMA 相関**: 結合行列 [X_cc (15), Y_sub,oc (79)]、窓 504・半減期 120・winsorize ±3σ は現行値を踏襲。当日行除外（`all_returns[window_start:current_index]`）の規約は実験コードでも厳守。**mh_blend 対応**: h ∈ {1, 3, 5} それぞれについて、本番と同様に horizon 別ターゲット（h 日 oc）で共通入力を再構成して推定する（`_prepare_common_inputs(horizon=h)` と同じ構造）
3. **縮約**: 94 次元 / 有効観測 ~504 は 17 次元時より推定誤差が大きいため、`alpha_yy` を 0.50 から引き上げ（候補 0.70、感度分析 0.50/0.60/0.70/0.80）する。`alpha_xx`/`alpha_yx`/`ρ` は本番値固定（探索しない = 過学習ガード）。
4. **事前分布**:
   - **PCA 事前分布 B_pca**: 事前部分空間 V0 を 94 次元に拡張。v1（グローバル）・v2（国スプレッド）は機械的に拡張可能。v3–v6 の感応度ラベルは **サブセクターが親セクターのラベルを継承**（クロスマッピングのサブセクターは構成銘柄加重平均）して生成する。`tickers.py` の `SENSITIVITY_LABELS` は本番正本のため**変更せず**、サブセクター用ラベルは research 側で生成。
   - **セクター事前分布 M_sector**: 79×15 の「米国セクター ↔ サブセクター」対応行列を構築。現行 `_SECTOR_MAPPING_STRUCTURE`（15→17 対応）を出発点に、各 JP セクターの下にぶら下がるサブセクターへ等分で展開し列正規化する。
   - **ベースライン相関 c_full**: サブセクター版も **2010–2014 固定**で `compute_baseline_correlation` に相当する計算を行う。先頭 1260 行フォールバックが発動する構成は作らない（不変条件 2）。
5. **確信度調整**: サブセクター空間では行わない（後段 §5.2 の集約後に既存の 17 次元パイプラインで実施。二重適用を避ける）。

### 5.2 集約・合成

```
ẑ_agg   = A ẑ_sub              （17 次元。A は §3.2、当日有効列のみで再正規化）
ẑ_blend,s = (1 − w_s)·ẑ_direct,s + w_s·ẑ_agg,s     （w_s = w_max × cov_s）
```

- 非カバーセクター（行が空）は ẑ_direct のみ。
- ẑ_blend を既存パイプラインの「ẑ_J」の位置に代入し、以降（確信度調整→逆標準化→gap 調整→v2 分布→ポートフォリオ）は無変更で通す。
- **実装上の差し込み点**: `compute_blp_signal` の ẑ_hat_j_t1 算出直後にフックする形を research 側サブクラスで実現する。本番クラスには手を入れない。

### 5.3 共分散側の精緻化（第1版では見送り）

`Omega_gap`（17×17）の精緻化（`A Ω_sub Aᵀ` のブレンド等）は理論上可能だが、v2 分布計算（`tools/research/compute_gap_adjusted_distribution.py` + on-demand フォールバック）の改変を伴うため **本実験のスコープ外** とし、平均側（mu）の改善が採用された場合の次フェーズ候補とする。

なお第1版には以下の **既知の近似** が残る（許容するがレポートに明記する）: ブレンド後の ẑ に適用する確信度調整の分散 `pred_var` は direct モデルの `Σ_Y|X` 由来であり、blend 推定量の真の分散とは一致しない。コード上 `omega_raw` は `Sigma` ブロックと `B_struct` のみから構成される（`core/gap_adjustment.py::_omega_from_blp_res`）ため、blend が ẑ のみを変更する限り `omega_gap` は不変であり、分布の一貫性は保たれる。blend 対応分散（`(1−w)² Σ_direct + w² A Σ_sub|X Aᵀ` 等）への置き換えは将来候補とする。

---

## 6. 実装計画（フェーズ分割）

配置規約: 実験モジュールは `src/research/experiments/subsector/`、実験スクリプトは `src/research/scripts/experiments/`、本番パス `src/leadlag/` には実験コードを入れない。`python3 -c` のインライン実行禁止・長時間コマンドはタイムアウト付き（`docs/スタック再発防止策.md`）。

実験 config（`configs/research/subsector_experiment.yaml`、研究用・本番 `configs/production/` には入れない）の素案:

```yaml
__base__: ../production/production.yaml  # 本番設定を継承し差分のみ上書き（読込時は必ず copy.deepcopy）
subsector:                                # pydantic スキーマ未登録の研究用ブロック
  taxonomy_file: configs/taxonomy_subsectors.yaml
  mapping_file: configs/research/subsector_mapping_generated.yaml
  panel_oc_path: var/research/subsector/panel_subsector_oc.parquet   # 推定用（open→close）
  panel_cc_path: var/research/subsector/panel_subsector_cc.parquet   # 参考診断用
  mask_path: var/research/subsector/mask_subsector.parquet
  alpha_yy: 0.70         # サブセクター BLPX の YY 縮約（§5.1-3、感度分析対象）
  w_max: 0.5             # 合成重み上限（§3.3、感度分析対象）
  basket_weighting: market_cap   # バスケット構成重み（§4.2。感度比較: equal / market_cap_static）
  agg_weighting: market_cap      # ETF 集約重み（§3.2。感度比較: equal / market_cap_static）
  min_valid_members: 2   # バスケット有効化の最少有効銘柄数（§4.2）
  min_member_ratio: 0.5  # バスケット有効化の最少有効比率（§4.2）
```

注: `subsector` ブロックは研究用であり、`build_app_config_from_dict` に投入する cfg からは除外してモデル側にのみ渡す。extra keys の扱いは `src/leadlag/config/schemas.py` の model_config を実装時に確認すること。

各 Phase は「目的 → 作業項目 → 成果物 → ゲート（次へ進む条件）」の構造で管理する。

---

### Phase 0: データ・マッピング構築（品質ゲート付き）

**目的**: 以降の全フェーズの土台となる (a) 銘柄→TOPIX-17 マッピング、(b) サブセクター oc（推定用）/ cc（参考）リターン時系列を構築し、品質を定量的にゲートする。

**作業項目**:

1. **マッピング構築**（`src/research/experiments/subsector/taxonomy_map.py`）
   - 入力: `configs/taxonomy_subsectors.yaml` + JPX 業種コードマスタ（JPX 公表の上場会社一覧から 33 業種コードを取得。手動ダウンロードし `var/research/subsector/jpx_master/` に保管・取得日を記録）
   - 33 業種 → TOPIX-17 のロールアップ表を同モジュール内の固定テーブルとして定義（手動個別指定は禁止、§3.5）
   - 出力: `configs/research/subsector_mapping_generated.yaml`（銘柄 × {subsector, topix17, 33業種コード}）+ 検証レポート（サブセクター × 17 セクターのクロス集計表・非カバーセクター一覧・`cov_s` 一覧）
   - **全射チェック**: 17 セクター全てに ≥1 個の有効サブセクターが写ること。未カバー時は当該セクター限定で最小限のサブセクターを taxonomy に追加（§3.4 の判断記録に従う。全 TOPIX 拡張は行わない）
2. **個別株データ取得**（`src/research/experiments/subsector/panel.py` の download 部）
   - yfinance で 498 銘柄の日次 OHLC（Adj Close 含む）を 2010-01-01 以降取得。既存プロバイダ（`src/leadlag/data/providers/yfinance_provider.py`）・キャッシュ機構を流用
   - ハング対策（`docs/スタック再発防止策.md` P2 準拠）: ティッカー単位タイムアウト・チャンク分割ダウンロード・再試行上限付き・途中保存
   - データ検証: ティッカー別欠損率、連続ゼロリターン（停滞・データ詰まり検出）、Adj Close/Close 比の異常ジャンプ（分割調整ミス検出）。欠損処理は `preprocess_data` の規約（side 中央値補間・有効ティッカー 50% 未満の日はスキップ）を準拠適用
3. **パネル構築**（同 `panel.py` の build 部）
   - アベイラビリティマスク（上場日以降かつデータ有効日のみ available）→ 時価総額加重バスケット 79 系列（§4.2、重み原料の PIT 時価総額は §4.4）
   - 取引カレンダーは既存 `df_exec` の index を正本としてアライン（日本営業日カレンダーの第二正本を作らない）
   - 出力: `var/research/subsector/panel_subsector_oc.parquet` + `panel_subsector_cc.parquet` + `mask_subsector.parquet` + 構築ログ（欠損率・除外銘柄・無効日一覧）
4. **品質検証**（`experiment_subsector_data_build.py` の report 部）
   - バスケット時価総額加重系列 vs 対応 TOPIX-17 ETF 系列の相関（全期間 + 年次分解。等加重版も参考計算）
   - セクター別 `cov_s` 計測（**時価総額ベースが主指標**: 分子は taxonomy カバー銘柄の cap 合計、分母は JPX 業種別構成データまたはセクター ETF 構成銘柄から構築するセクター全体 cap。カウントベースは参考併記）

**ゲート（全て PASS で Phase 1 へ）**:
(1) 未分類銘柄 0 件
(2) 全射性: 17 セクター全てに ≥1 個の有効サブセクター
(3) バスケット vs ETF 相関 ρ ≥ 0.7（未満のセクターは要調査・除外判断を記録）
(4) セクター別 `cov_s` が計測・記録済み
(5) 2010–2014 ベースライン期間の有効サブセクター数 ≥ 60（目安。ベースライン相関 `c_full` の推定安定性確保のため。未達なら最少銘柄数ルールや構成を再検討）

**成果物**: `configs/research/subsector_mapping_generated.yaml`、パネル/マスク parquet、`reports/subsector_refinement/phase0_data/report.md`

**想定リスク**: yfinance レート制限・ハング（タイムアウト + 分割取得で対応）、新規上場銘柄の欠損期間（マスクで処理済みのはず → ログで確認）、JPX マスタの形式変更（取得時スナップショット保管で再現性確保）。

### Phase 1: オフライン IC 評価（安価な先行検証）

**目的**: フルバックテスト（計算コスト大）の前に「サブセクター経由で 17 次元予測が改善するか」をシグナルレベルで検証する。負けたら早期打ち切りし、Phase 2 以降のコストを節約する。

**作業項目**:

1. **予測系列の生成**（`src/research/scripts/experiments/experiment_subsector_ic.py`）
   - 2015-01-05〜最新の全営業日について、以下の 3 系列を日次計算・保存（`var/research/subsector/ic_series/`）:
     - (a) `ẑ_direct`: 現行 17 セクター Residual-BLPX の予測（`compute_blp_signal(is_residual=True)` の ẑ_hat_j_t1 相当）
     - (b) `ẑ_agg = A ẑ_sub`: サブセクター BLPX（§5.1）の予測を §3.2 の集約行列で 17 次元化
     - (c) `ẑ_blend = (1−w_s)·(a) + w_s·(b)`（w_max=0.5、`cov_s` は Phase 0 の計測値）
   - 両モデルとも strictly historical（当日行除外・2010–2014 ベースライン固定）であることをコードレビューで確認（不変条件 1, 2）
2. **実現ターゲットの構成**: 17 セクター ETF の翌営業日 9:10→大引けリターン（`compute_jp_target_returns` と同定義の実績値を `df_exec` から再構成）。参考系列として cc リターンとの IC も算出
3. **評価統計**:
   - 日次断面 IC（Pearson）/ Rank IC（Spearman）の系列平均・t 統計量（Newey-West HAC 補正）・block bootstrap 95% CI
   - **tail 指標（必須）**: 戦略が実際に使うのは上下 5 銘柄のみであり、全断面 IC の改善が tail に載らない可能性がある（§9）。予測上位 5 − 下位 5 の日次スプレッド（実現値）の系列、(c) vs (a) での tail 内ランク一致率・tail 限定 Rank IC を算出し、tail でも改善していることを確認する
   - (c) vs (a) の日次 IC 差の paired 検定（one-sided）
   - **セクター別 IC 分解**（どのセクターで改善/悪化しているか）・**年次分解**（改善の時系列偏り検出 — §4.3 のサバイバーシップ論点との整合確認）
   - **メカニズム検証**: セクター別 IC 改善幅と `cov_s` の相関（カバレッジが高いセクターで改善が大きい = 仮説 H1 と整合）
   - 中間生成物の健全性: サブセクター単位 IC の分布（どのサブセクターが効いているか / 足を引っ張っているか）
4. **サブ期間検証**: 直近 5 年（2021–2026、メンバーシップ誤差が小さい期間）で効果が保つか再確認（§4.3 緩和策）

**ゲート**: `ẑ_blend` の平均 Rank IC が `ẑ_direct` を one-sided p < 0.05 で上回る。**未達なら Phase 2 へ進まず**、不採用記録化（`reports/subsector_refinement/` + `docs/experiment_graveyard.md`）して終了

**成果物**: `reports/subsector_refinement/phase1_ic/`（summary md + IC 系列 csv + セクター別/年次分解の図表）

### Phase 2: フル V2 バックテスト

**目的**: 取引層（mu_over_sigma ランキング → L/S 5 銘柄選択 → MinVar → RuleD）・コスト・オーバーレイ（ml_overlay / cs_overlay / mh_blend）全込みで net Sharpe を baseline と同一条件比較する。

**配線設計（重要な実装判断）**: `BacktestEngine.run_v2_backtest()` は内部で `ProductionBLPXModel(run_cfg.blpx)` を直接生成し（`src/leadlag/execution/backtester.py` L453）、外部からモデルを注入する口がない。また日次の分布解決はファイルキャッシュ（`gap_input_dir` 内の `mu_gap`/`omega_gap`）が主経路である。したがって以下の **Route A（研究用 gap ストア差し替え）** を採用する:

- **Route A（採用）**: 強化モデルで gap 調整済み分布を全日分事前計算して研究用ストアに書き、`run_v2_backtest(cfg, gap_input_dir=<研究ストア>)` を **無変更の本番コードで** 実行する。ファイルキャッシュ経路は検証済み（`validate_gap_matrices`）の主経路であり、baseline（本番ストア）との差分は mu_gap のみに限定できる
- **Route B（不採用）**: バックテスター内の `ProductionBLPXModel` シンボルをモンキーパッチして on-demand 計算させる方式は、`n_jobs > 1` では loky ワーカープロセスがモジュールを再 import するためパッチが伝播せず破綻する。逐次実行強制は計算コスト的に非現実的

**作業項目**:

1. **強化モデル**（`src/research/experiments/subsector/model.py`）: `ProductionBLPXModel` のサブクラス。`compute_blp_signal` をオーバーライドし、`signal_computer.py` が `__all__` で公開する既存ヘルパー（`_prepare_window_returns` → `_estimate_correlation` → `_solve_blp_coefficients` → `_compute_pca_prior` → `_get_sector_prior` → `_solve_tikhonov`）を同順で再利用してステップ 1–7 を再構成し、**ステップ 5（ẑ_hat_j_t1 算出）とステップ 6（確信度調整）の間に §5.2 の集約・ブレンドを挿入**する。コード複製を避け、本番クラス・本番モジュールは無変更
2. **研究用 gap ストア生成**（`src/research/scripts/experiments/experiment_subsector_gap_store.py`）: `tools/research/compute_gap_adjusted_distribution.py` の処理を研究用にミラーし、強化モデルで `mu_gap`/`omega_gap` を h ∈ {1, 3, 5}（本番 `mh_blend` 設定 `mh_horizons`/`mh_weights` に合わせる）で全バックテスト日分生成 → `var/research/subsector/gap_store.sqlite`（**本番 `var/live/pipeline_data/` とは物理分離**）
   - **horizon 別推定**: h>1 ではターゲット自体が h 日 oc に変わり共通入力も再構成される（`_prepare_common_inputs(horizon=h)`）。サブセクター側も horizon 別の h 日 oc ターゲットで推定する（§5.1-2）。h=1 のサブセクター結果を h=3,5 に流用してはならない
   - **整合診断**: 第1版では共分散側は不変のはず（§5.3）→ 研究ストアと本番ストアの `omega_gap` 差分が ≈0 であることを全日確認。有意差があれば意図せぬ共分散変更として停止・調査
   - 生成前に 100 日サンプルでタイミング計測し、全期間の所要時間を見積もってから本実行（タイムアウト・途中再開可能な設計）
3. **バックテスト実行**（`experiment_subsector_backtest.py`）:
   - baseline: 本番 gap ストア（`var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite`）+ `configs/production/production.yaml`（`copy.deepcopy`）
   - treatment: 研究 gap ストア + 同一 cfg（`copy.deepcopy` — shallow copy 禁止の既知の落とし穴）
   - 同一 `df_exec` スナップショット・同一コストパラメータ・同一オーバーレイ設定（ml_overlay phase2_8 / cs_overlay / mh_blend は本番通り有効のまま）で A/B 比較
   - CLI 相当: `python3 -m leadlag.cli backtest --config configs/production/production.yaml --start-date 2015-01-05 --gap-dir <各ストア>` をスクリプト経由・タイムアウト付きで実行
4. **テスト**（test-gen 方針、research 側に配置: `src/research/experiments/subsector/tests/`）:
   - **配線パリティテスト（最重要）**: `w_max=0`（ブレンド無効化）で強化モデルの出力 `mu_gap`/`omega_gap` が本番モデルと全 h でビット一致すること — サブクラスの再構成（ステップ 5a の非対称伝播・gap 係数の符号分岐等のインライン処理を含む）が本番と厳密に等価であることの証明。これが通るまで Phase 2 本実行しない
   - 集約行列 A の凸性（行和 = 1 または空行）・非カバー行の `w_s=0` 縮退・当日無効列の再正規化
   - NaN/Inf が ẑ_blend に伝播しないこと（`no_nan_inf_in_signals` 監査と整合）
   - 相関窓・β 窓の当日行除外（`all_returns[window_start:current_index]` 規約のサブセクター版）
   - ベースライン期間分離（サブセクター版 `c_full` が 2010–2014 のみで計算されること）
   - 完了後は既存全テスト `bash scripts/run_tests_parallel.sh` がグリーンであること（不変条件 3）

**指標**: net Sharpe（主指標）/ gross Sharpe / 最大 DD / ターンオーバー / フォールバック発動率 / コスト内訳（slippage・financing・borrow・reverse 分解）/ 年次別 Sharpe。加えて **baseline との同一条件性チェック**（取引日数・フォールバック発動日の一致、ウェイト差分の分布）を報告

**成果物**: `reports/subsector_refinement/phase2_backtest/`（backtest-report スキル形式の md + 日次系列 csv）

**想定リスク**: (a) gap ストア生成の計算コスト（両モデル × ~2,900 日 × 3 horizon → 並列化と途中再開で対応）(b) h=3,5 行列の生成漏れ（バックテスト前に `matrices/mu_gap_h{h}_{date}` の全日存在を検証）(c) 研究ストアの検証失敗が多発しフォールバック率が上昇する場合は、信号側の数値異常として Phase 1 へ差し戻し。

### Phase 3: ウォークフォワード + 過学習ガード

**目的**: 同一ヒストリー上での反復選択バイアスを定量化し、OOS 頑健性を確認する（AGENTS.md 過学習ガード必須要件）。

**作業項目**:

1. **年次 OOS 分割**: 2015–2026 の 12 年次区間、purge=61 日（相関窓 60+1）、embargo=5 日（先例: `src/research/scripts/experiments/experiment_a7_walkforward_dsr.py`）。本モデルは推定パラメータを持たず（事前分布・ベースライン相関は 2010–2014 固定）ため、WF は「年次別 OOS 成績分解 + 設定の再チューニング禁止」として機能させる。各区間で baseline / treatment を同一条件で再実行
2. **パラメータ感度分析**: 新規パラメータは原則 `w_max` 1 個（+ `alpha_yy` 引き上げを選ぶ場合は合計 2 個）
   - `w_max` ∈ {0.40, 0.50, 0.60}（±20% 摂動）+ 参考水準 {0.25, 0.75}
   - `alpha_yy` ∈ {0.50, 0.60, 0.70, 0.80}
   - **重み付けロバスト性**: バスケット構成・ETF 集約の重み付け 3 案（PIT 近似時価総額 / 固定スナップショット時価総額 / 等加重、§4.4）での結果の安定性を確認（チューニング対象ではなくロバスト性検証。DSR の n_trials には計上する）
   - 要件: ±20% 摂動で Sharpe 変動 < 20%。グリッド最良点は選ばず中央設定を本番候補に固定
3. **Deflated Sharpe Ratio**（Bailey & López de Prado 2014）: n_trials = 過去実験（`archive/experiments/` 約 30 本）+ `reports/` 配下の実験数 + 本実験バリアント数（w_max 3 + alpha_yy 4 + Phase 1 系列等 ≈ 20）で **n_trials ≈ 50 以上**を計上して補正。DSR ≥ 0.95 を目安
4. **逐次検定の扱い**: Phase 1 ゲート通過後に Phase 3 を実行する設計は逐次選択に相当するため、DSR 閾値の保守性（実効試行回数の過小/過大計上の方向性）をレポートで議論する。加えて **Phase 1 ゲート自体が 2015–2026 全期間の in-sample 評価である**点を明記し、実装時に「Phase 1 を 2015–2021 に限定し 2022–2026 を pure holdout として温存する」分割案を検討する（holdout は Phase 2/3 で初めて使用）

**判定基準**（experiment-design スキル準拠）: 全区間 Sharpe > 0（負区間 ≤ 2）/ baseline 対比勝率 ≥ 8/12 / ±20% 摂動で変動 < 20% / DSR ≥ 0.95

**成果物**: `reports/subsector_refinement/phase3_walkforward/`（年次表 + 感度ヒートマップ + DSR 計算書）

### Phase 4: 判定・記録・（採用時）本番昇格設計

**目的**: 全エビデンスを統合して採否を決定し、採用・不採用いずれの場合も再検証不要な形で記録を残す。

**判定**: 以下をすべて満たす場合のみ採用（experiment-design スキル / AGENTS.md 採用基準）:
(1) Pooled net Sharpe 改善 > 0.5 SE ≈ 0.01 (2) WF 勝率 ≥ 8/12 (3) DSR ≥ 0.95 (4) ±20% 摂動で Sharpe 変動 < 20% (5) フォールバック発動率の非悪化 (6) ターンオーバー非増大

**採用時の作業**:

1. **本番統合の設計 ADR** を `docs/decisions/` に作成（サブセクター層の本番昇格方式: 現行 `ProductionBLPXModel` へのフラグ追加 vs 別クラス本番登録。**フラグ OFF で現行挙動に即時ロールバック可能**であることを要件とする）
2. **日次運用への組込み**: サブセクターパネルの日次更新（前営業日大引けの個別株終値 → バスケット更新）を 6:30 JST の gap 分布バッチ（`scripts/batch/run_gap_distribution.sh`）の前提ステップに追加。`docs/日次運用手順書.md` 更新。パネル欠損時の挙動（direct のみで継続 = 現行挙動への自動縮退）を明文化
3. **シャドー運用**: `tools/validation/monitor_residual_blpx_shadow_performance.py` / `shadow_runs/` で新旧シグナルのライブ乖離を 1 ヶ月以上観測してから本番切替（AGENTS.md 改善ワークフロー 4）
4. **ドキュメント**: `configs/production/` 更新 + `docs/ARCHITECTURE.md` リファクタリング履歴追記 + `docs/モデル技術仕様書.md` への数式追加（§5.1/§5.2 の定式化）

**不採用時の作業**（experiment-design スキルのクリーンアップ手順通り）: レポートのみ `reports/subsector_refinement/` に保持、実験コード（`src/research/experiments/subsector/`・関連スクリプト）・中間データは破棄、`docs/experiment_graveyard.md` に 1 行サマリー追記

**いずれの場合も**: 最終レポートは backtest-report スキルの標準形式（Hypothesis / Methods / Results / Analysis / Conclusion / Files）で `reports/subsector_refinement/` に残す

---

## 7. 不変条件チェックリスト（AGENTS.md 準拠）

| # | 不変条件 | 本計画での担保 |
|:--|:---|:---|
| 1 | ルックアヘッド禁止 | サブセクターパネル・β・相関窓は全て strictly historical + 当日行除外。`ComplianceAuditor` / `v2_auditor` の監査項目は無効化しない。ただし **taxonomy メンバーシップは現時点スナップショット**（§4.3 の制約として明示・定量化） |
| 2 | ベースライン期間分離 | サブセクター版 c_full も 2010–2014 固定。先頭 1260 行フォールバック構成は作らない |
| 3 | テストを弱めない | 変更後は全テスト実行（並列版）。実験は `src/research/` 閉じ込めで既存テストに影響させない |
| 4 | 市場中立制約 | 下流（ポートフォリオ構築・RuleD・risk.py）を無変更にすることで構造的に保全 |
| 5 | ティッカー定義 | `tickers.py`（N_U=15, N_J=17, SENSITIVITY_LABELS）は変更しない。サブセクターユニバースは research config のみ |
| 6 | 前日 gap 行列コピー禁止 | gap 調整は従来通り 17 セクター層。本実験は gap データの扱いを変更しない |

---

## 8. エッジケース・リスク（edge-case-finder 観点）

| # | リスク | 重大度 | 対応 |
|:--|:---|:---|:---|
| 1 | 推定次元 94 / 窓 504 で相関推定誤差増大 | High | `alpha_yy` 引き上げ + 感度分析。Phase 1 IC で早期検出 |
| 2 | 新規上場銘柄の上場前 NaN がバスケットを歪める | High | アベイラビリティマスク + 最少有効銘柄数ルール（§4.2） |
| 3 | 非カバー / 低カバレッジセクターで集約値が不安定 | Medium | `w_s = w_max × cov_s` で direct に縮退。行再正規化で凸性維持 |
| 4 | サブセクター ≤3 銘柄（11 個）のバスケットノイズ | Medium | 最少銘柄数ルールで無効化 → direct フォールバック |
| 5 | yfinance 個別株の欠損・ハング（IJR 型欠損の 498 銘柄版） | High | ティッカー別欠損検査 + タイムアウト + 再試行（`docs/スタック再発防止策.md` P2 準拠）。ダウンロードは日次本番とは分離したバッチ |
| 6 | サバイバーシップ / メンバーシップルックアヘッド | High | §4.3 の緩和策 + レポートへの明記 |
| 7 | ẑ_sub の NaN が合成へ伝播 | Medium | 集約時マスク処理 + ẑ_blend 出力の NaN/Inf アサート（監査の `no_nan_inf_in_signals` に整合） |
| 8 | クロスマッピング銘柄の二重計上 | Medium | 銘柄レベル集約定義（§3.2）により構造的に排除（各銘柄は 1 行にのみ寄与） |
| 9 | 計算コスト | Low | 94×94 EWMA 相関 × ~2,900 日でも既存バックテストと同オーダー。Phase 1 で実測 |
| 10 | config shallow copy で direct/sub 両モデルが同一設定化 | Medium | 既知の落とし穴に明記済みの通り `copy.deepcopy` 徹底 |
| 11 | PIT 時価総額の近似誤差（増資・自社株買いドリフト、浮動株調整なし、現在株式数の過去投影） | Medium | §4.4 の感度比較（PIT 近似 / 固定 / 等加重）で影響を定量化。バスケット・集約ともセクター内相対値なので二次的影響に留まる想定 |
| 12 | 時価総額加重で小バスケットが最大銘柄に支配される（実質単一銘柄化） | Medium | §4.2 の集中度ログ（最大重み > 70% 等）で診断。最少有効銘柄数ルールと併用 |

---

## 9. 評価指標の約束事（本実験での適用）

- 主指標: **net Sharpe**（コスト後）。gross/net 両報告、コスト内訳分解
- Phase 1（IC）と Phase 2（バックテスト）の乖離に注意: IC 改善があっても取引層（ランキング上位/下位 5 銘柄の選択）に効果が載らないケースがありうる（ランキングベース戦略の非線形性）。レポートでは両者の関係を必ず分析する
- 「改善なし」も価値ある結論として `reports/subsector_refinement/` に形式通り記録

---

## 10. スコープ外・将来拡張

- **米国側の細分化**（SMH/IGV/XBI 等のサブ産業 ETF を X に追加）: サブセクター化の利益の半分は米国側マッピングの精緻化にある可能性が高いが、ユニバース変更（不変条件 5）を伴うため本実験では分離。JP 側のみで効果が確認できた場合の次フェーズ
- **サブセクター版 Omega_gap**（共分散精緻化）: §5.3
- **真の PIT 時価総額（発行済株式数の履歴・浮動株比率）の導入**（Bloomberg/FactSet 等の有償データ導入時。第1版は §4.4 の近似で実施）
- **PIT メンバーシップ再構築**（同上）
- **サブセクター分類の TOPIX 全体への拡張**: 第1版では行わない判断と理由・発動条件は §3.4 を参照
- **サブセクター 9:10 価格の活用**（498 銘柄の寄付気配を使った gap 調整の精緻化）: データ・運用負荷が大きく、当面見送り

## 11. オープンクエスチョン（実験前に要確認）

1. JPX 業種コードマスタの取得経路とライセンス（手動ダウンロードで足りるか）
2. 498 銘柄 × 2010 年以降の yfinance 実データ品質（欠損率・分割調整の妥当性）の実測 — Phase 0 ゲートで判定
3. サブセクター版 v3–v6 感応度ラベルの継承方法（等分継承 vs 加重平均）— 影響は小さいと予想（2026-07 の検証で感応度ラベル差異の Sharpe 影響 < 0.01 の実績）だが Phase 1 で両案比較
4. `w_max` の既定値 0.5 の妥当性（0.25/0.5/0.75 の 3 水準を感度分析で確認し、本番候補は中央値に固定）
5. サブセクター oc（open ベース）と ETF oc（直近期は真の 9:10 mid）の定義差が推定相関に与える影響の実測 — §4.1 のロバスト性確認で判定
6. 79×15 への展開でセクター事前分布 `M_sector` が希薄化する影響（`lambda_sector=0.60` は強い事前重み。サブセクター空間で事前分布が過度に薄まり効果が変わらないか）— Phase 1 で `B_struct_sub` の事前分布寄与を診断
7. `cov_s` は §4.4 の PIT 時価総額パネルから時変計測（月次更新）を主とし、静的スナップショットは感度比較。ただし taxonomy メンバーシップ自体は現時点スナップショットのまま（§4.3 の制約は残る）
8. §4.4 の PIT 時価総額近似の妥当性（分割遡及調整の正しさ・増資ドリフトの大きさ）は Phase 0 でサンプル検証（既知の株式数変動イベント銘柄で突合）
