# 2015〜2017 JPX33時価総額ウェイトのPIT訂正影響

## 判定

**2017年末ウェイトを2015〜2017年へ遡及する比較は時点整合性を満たさない。** 前年12月のJPX業種時価総額を使い、各年最初の日本取引日を除外して翌日から有効とした修正版を、同じ3年期間で比較した。成績は過去を既に確認した回顧診断で、本番採用や未使用OOSの根拠にはしない。

- 対象日は2015-01-06〜2017-12-29の714日。2015〜2017年の各年最初の取引日 2015-01-05, 2016-01-04, 2017-01-04 は、JPXが第1営業日13時以降に前年末資料を公表するため比較から除いた。
- ターゲットはTOPIX残差化した始値→大引けリターン。各日のBLPX入力窓は当日を除外し、c_fullは2010〜2014固定。9:10→大引け成績ではない。
- fixed_2017 は従前の33業種スコアを2017年12月市場時価総額で17ETF priorに写像。annual PITは2015に2014、2016に2015、2017に2016の年末業種時価総額を使用した。

## 予測指標

| 変種 | paired日数 | mean Rank IC | mean MAE (bp) | mean RMSE (bp) |
|---|---:|---:|---:|---:|
| current_17 | 714 | +0.0988 | 65.77 | 86.40 |
| legacy_33_fixed_2017 | 714 | +0.0999 | 65.73 | 86.35 |
| legacy_33_pit_cap | 714 | +0.0998 | 65.73 | 86.35 |

### 2017固定ウェイトと年次PITウェイトの差

| 比較 | ΔRank IC 95% block CI | ΔMAE bp 95% block CI | ΔRank IC平均 | ΔMAE平均 (bp) | 日数 |
|---|---:|---:|---:|---:|---:|
| fixed_2017_vs_pit | [-0.0003, +0.0001] | [-0.00, +0.00] | -0.0000 | -0.00 | 714 |
| current_vs_fixed_2017 | [-0.0007, +0.0027] | [-0.06, -0.03] | +0.0011 | -0.04 | 714 |
| current_vs_pit | [-0.0008, +0.0026] | [-0.06, -0.03] | +0.0011 | -0.04 | 714 |

CIは20営業日non-circular moving-block bootstrap、5,000回、seed 20260924。Δは右辺variant−左辺variant。正のMAE差は誤差悪化を表す。

## 時点整合・解釈

PIT年次更新は固定2017表より情報整合的だが、JPXの年末業種時価総額は浮動株調整ウェイトではない。2015〜2021はFirst Section、2022年以降はPrimeへ公表対象市場が変わる。また当研究の比較区間は既知データであり、修正後でも将来OOSではない。
この検証は固定2017ウェイトが作る入力priorの差を実際の3年予測指標で測った。ここで示すのは同じ実績期間内での訂正影響であり、訂正後モデルが一般に優れるという結論ではない。

## 再現情報

- スクリプト: `src/research/scripts/experiments/experiment_jpx33_pitcap_correction_2015_2017_20260924.py`。prior mapping: `configs/research/jpx33_sensitivity_prior_2017.yaml`。年次時価総額: `configs/research/jpx33_yearend_market_cap_2014_2025.csv` と `configs/research/jpx33_yearend_market_cap_2014_2025.sources.json`。
- df_exec SHA-256: `f826fa8eb58ce35a17d29c8a13f38513c48daf2669c9ff32d9b9a67bd48d81fb`。BLPX parameter set: `phase1d_optimized_residual_blpx_production`。本番設定・本番コードは変更なし。
