# サブセクター細分化実験の感応度ラベル・分類分析

## 概要

日米リードラグ戦略のサブセクター細分化実験（79次元サブセクター空間→17 TOPIX ETF集約）において、ICがdirect 17-dim BLPXを大幅に下回る問題（mean IC -0.025 vs +0.159）の根本原因を分析するための資料。

## 現在のサブセクター分類（79サブセクター）

### 集約行列Aの構造

- **形状**: 17 (ETF) × 79 (サブセクター)
- **集約方式**: value-weighted（時価総額加重）
- **銘柄数**: 494銘柄（494銘柄バージョン）/ 1620銘柄（拡張バージョン）

### 79サブセクター一覧

1. Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材)
2. Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送)
3. Airlines & Airport Infrastructure (航空・空港インフラ)
4. Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売)
5. Auto Parts & Systems (自動車部品・電装品・内装)
6. Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材)
7. Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ)
8. Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス)
9. City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給)
10. Comprehensive Real Estate Developers (総合不動産・ビルデベロッパー)
11. Confectionery, Bakery & Snacks (製菓・スナック・製パン)
12. Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械)
13. Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援)
14. Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融)
15. Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール)
16. Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品)
17. Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア)
18. Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融)
19. Cosmetics & Personal Care (化粧品・トイレタリー・日用品)
20. Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品)
21. Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設)
22. Digital, Retail & Specialized Banks (ネット・流通・専門銀行)
23. Drugstores & Dispensing (ドラッグストア・調剤薬局)
24. Electric Power Utilities (電力・発電・送配電)
25. Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品)
26. Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発)
27. Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS)
28. General Contractors (ゼネコン・総合建設)
29. General Trading Companies - Sogo Shosha (総合商社)
30. Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材)
31. Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー)
32. Global Major Innovative Pharma (グローバル新薬・大型創薬)
33. Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材)
34. Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発)
35. HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生)
36. Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池)
37. Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ)
38. Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材)
39. IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通)
40. Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装)
41. Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング)
42. Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板)
43. Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網)
44. JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網)
45. Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通)
46. Life & Non-Life Insurance (生命保険・損害保険)
47. Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング)
48. Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器)
49. Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀)
50. Marine Shipping (外航海運・海上輸送)
51. Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学)
52. Megabanks & Trust Banks (メガバンク・ゆうちょ・信託)
53. Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料)
54. Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器)
55. Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ)
56. Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ)
57. Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント)
58. Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料)
59. Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷)
60. Passenger Cars, Commercial Vehicles & Motorcycles (乗用車・完成車・商用トラック・二輪)
61. Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事)
62. Regional & Provincial Banking Groups (地域ブロック中核地銀)
63. Restaurants & Food Service (外食・フードサービス)
64. Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品)
65. Securities, Exchanges & Investment (証券・取引所・投資育成)
66. Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援)
67. Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル)
68. Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE)
69. Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸)
70. Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材)
71. Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック)
72. Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品)
73. Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア)
74. Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル)
75. Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター)
76. Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通)
77. Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー)
78. Tires & Rubber Products (タイヤ・ゴム製品)
79. Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流)

## 17 TOPIX ETF (ターゲット)

1. 1617.T 食品
2. 1618.T エネルギー資源
3. 1619.T 建設・資材
4. 1620.T 素材・化学
5. 1621.T 医薬品
6. 1622.T 自動車・輸送機
7. 1623.T 鉄鋼・非鉄
8. 1624.T 機械
9. 1625.T 電機・精密
10. 1626.T 情報通信・サービス
11. 1627.T 電力・ガス
12. 1628.T 運輸・物流
13. 1629.T 商社・卸売
14. 1630.T 小売
15. 1631.T 銀行
16. 1632.T 金融（除く銀行）
17. 1633.T 不動産

## 感応度ラベル（v3~v6）の定義

### 感応度ラベルの意味

- **w3**: 成長感応度
- **w4**: テクノロジー感応度
- **w5**: コモディティ感応度
- **w6**: 金利感応度

### 現在の17 ETF感応度ラベル（本番設定）

```python
# 17 TOPIX ETFs
"1617.T": _sens(-1.0, -0.6, -0.3, -0.3),  # 食品
"1618.T": _sens(0.3, 0.3, 1.0, 1.0),       # エネルギー資源
"1619.T": _sens(0.6, 0.3, 0.0, 0.3),       # 建設・資材
"1620.T": _sens(1.0, 0.6, 0.3, 0.6),       # 素材・化学
"1621.T": _sens(-1.0, -0.3, 0.0, -0.3),    # 医薬品
"1622.T": _sens(1.0, 1.0, -0.3, 0.0),       # 自動車・輸送機
"1623.T": _sens(1.0, 0.6, 0.3, 0.6),       # 鉄鋼・非鉄
"1624.T": _sens(1.0, 1.0, 0.0, 0.3),       # 機械
"1625.T": _sens(1.0, 1.0, 0.0, -0.3),      # 電機・精密
"1626.T": _sens(-0.3, -0.3, 0.0, -0.3),    # 情報通信・サービス
"1627.T": _sens(-1.0, -1.0, -1.0, -1.0),   # 電力・ガス
"1628.T": _sens(-0.3, -0.3, 0.0, -0.3),    # 運輸・物流
"1629.T": _sens(0.6, 1.0, 0.6, 1.0),       # 商社・卸売
"1630.T": _sens(-0.6, -0.6, -0.3, -0.6),   # 小売
"1631.T": _sens(1.0, 0.3, 0.0, 0.3),       # 銀行
"1632.T": _sens(0.6, 0.0, 0.0, 0.0),       # 金融（除く銀行）
"1633.T": _sens(0.6, -1.0, 0.0, 0.3),      # 不動産
```

### サブセクターへの感応度ラベル割り当て方法

#### 方法1: ドミナントETF継承（実験で使用）

各サブセクターに対して、集約行列Aで最大重みを持つETFの感応度ラベルを継承。

- **問題点**: ドミナント重みが非常に低い（平均0.20、中央値0.11）
- 64/79のサブセクターでドミナント重み < 0.3
- サブセクターが複数ETFに分散しており、単一ETFのラベル継承が不適切

#### 方法2: ヒューリスティック割り当て（検証済み）

サブセクターのドメイン知識に基づいてヒューリスティックに感応度ラベルを割り当て。

- **結果**: ICはドミナントETF継承とほぼ同じ（mean IC -0.025 vs -0.025）
- 感応度ラベルの調整はIC悪化の主因ではないことが判明

## IC評価結果

### Direct 17-dim BLPX（本番・ベースライン）

- mean IC (per-ETF): +0.159
- median IC (per-ETF): +0.162

### サブセクター実験（79次元→17ETF集約）

#### ドミナントETF継承方式

- mean IC (per-ETF): -0.025
- median IC (per-ETF): -0.020
- mean daily IC (cross-sectional): -0.070

#### ヒューリスティック感応度ラベル方式

- mean IC (per-ETF): -0.025
- median IC (per-ETF): -0.020
- mean daily IC (cross-sectional): -0.070

#### 拡張版（1620銘柄）

- mean IC (per-ETF): -0.096
- median IC (per-ETF): -0.093
- mean daily IC (cross-sectional): -0.058

## 品質ゲート結果

### Phase 0: バスケットvs ETF相関

- **ゲート**: 74/79以上のサブセクターで相関 >= 0.7
- **実績**: 74/79のサブセクターで相関 < 0.7（未達）
- **median相関**: 0.54
- **min相関**: 0.27 (494銘柄版) / 0.36 (1620銘柄版)

### Phase 1: サブセクターパネル品質

- サブセクターバスケットと対応ETFの相関が低いことが確認
- 増加した次元性によるノイズが懸念される

## 問題点の整理

### 1. 感応度ラベルの問題

- ドミナントETF継承方式では、単一ETFの重みが低いためラベルの代表性が不十分
- ヒューリスティック方式でもIC改善なし → 感応度ラベルは主因ではない

### 2. サブセクター分類の問題

- 79サブセクターが17ETFに過度に分散している可能性
- 各サブセクターが複数ETFにまたがり、純粋なセクター特性が損なわれている可能性
- バスケット構成銘柄の代表性が不十分（ETFとの相関が低い）

### 3. 次元増大の問題

- 79次元空間での推定不安定性
- パラメータ推定のノイズ増加
- 集約時の情報損失

### 4. 集約行列Aの問題

- value-weighted集約が最適かどうか
- 各サブセクターのETF純度が低い
- クロスETF contaminationが高い可能性

## 分析への提案

上位モデルによる分析において、以下の観点からの検討を提案：

1. **サブセクター分類の再設計**
   - 79サブセクターを減らす（例: 30-40サブセクター）
   - 各サブセクターを単一ETFに純粋に紐づける
   - ETF境界を尊重した分類

2. **感応度ラベルの再設計**
   - サブセクターの経済特性に基づく感応度ラベルの最適化
   - 加重平均ではなく、サブセクター固有の感応度プロファイルの推定
   - ドメイン知識とデータ駆動アプローチの統合

3. **集約行列Aの最適化**
   - value-weighted以外の集約方式の検討
   - 純度最大化を目的とした重み付け
   - ETFへの純粋なマッピング

4. **次元削減戦略**
   - 有効なサブセクターの選別
   - クラスタリングによる類似サブセクターの統合
   - 階層的サブセクター構造の導入

## データソース

- 集約行列: `configs/research/subsector_aggregation_vw.yaml`
- 拡張版集約行列: `configs/research/subsector_mapping_expanded.yaml`
- ヒューリスティック感応度ラベル: `configs/research/subsector_sensitivity_labels_heuristic.yaml`
- 本番感応度ラベル: `src/leadlag/data/tickers.py` (SENSITIVITY_LABELS)
- IC評価結果: `reports/subsector_refinement/phase2_blpx/subsector_ic_report.md`
