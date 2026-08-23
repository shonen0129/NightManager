# Phase 0: Subsector Panel Quality Report
**Date**: 2026-08-23
**Panel files**: /Users/shonen/leadlag/var/research/subsector/panel_subsector_oc.parquet
**Rows (dates)**: 4168
**Subsectors**: 79
**Overall valid observation ratio**: 0.9407

## Phase 0 Gates
- (1) Unmapped tickers in canonical mapping: 0 (need 0) — see excluded_unmapped section
- (2) All 17 sectors covered: True (need True)
- (3) Basket vs ETF correlation >= 0.7: 74 subsectors below threshold; min rho = 0.2653; median = 0.5122 (need min >= 0.7)
- (4) Sector coverage measured and recorded: True
- (5) Baseline (2010-2014) subsectors with >=126 valid days: 78 (need >= 60)

## Aggregation matrix A validation
- Row sums close to 1.0 or empty: True
- Empty rows (non-covered sectors): 0
- Shape: (17, 79)

## Subsectors with rho < 0.7
- General Contractors (ゼネコン・総合建設) vs 1619.T: rho = 0.5681
- Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材) vs 1619.T: rho = 0.5742
- Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事) vs 1619.T: rho = 0.5446
- Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ) vs 1619.T: rho = 0.4864
- Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材) vs 1620.T: rho = 0.4757
- Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル) vs 1620.T: rho = 0.5144
- Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材) vs 1620.T: rho = 0.5022
- Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料) vs 1620.T: rho = 0.4082
- Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル) vs 1620.T: rho = 0.4752
- Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材) vs 1619.T: rho = 0.5276
- Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷) vs 1620.T: rho = 0.3872
- Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板) vs 1623.T: rho = 0.5874
- Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料) vs 1623.T: rho = 0.5959
- Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ) vs 1623.T: rho = 0.6129
- Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装) vs 1624.T: rho = 0.5122
- Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器) vs 1624.T: rho = 0.4782
- Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械) vs 1624.T: rho = 0.5586
- Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング) vs 1624.T: rho = 0.5471
- Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池) vs 1625.T: rho = 0.5716
- Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE) vs 1625.T: rho = 0.5492
- Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品) vs 1625.T: rho = 0.6036
- Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ) vs 1625.T: rho = 0.5371
- Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学) vs 1625.T: rho = 0.4129
- Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器) vs 1625.T: rho = 0.5292
- Auto Parts & Systems (自動車部品・電装品・内装) vs 1622.T: rho = 0.6730
- Tires & Rubber Products (タイヤ・ゴム製品) vs 1622.T: rho = 0.5803
- Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック) vs 1621.T: rho = 0.5291
- Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品) vs 1621.T: rho = 0.5287
- Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ) vs 1617.T: rho = 0.5644
- Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品) vs 1617.T: rho = 0.4643
- Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品) vs 1617.T: rho = 0.5513
- Confectionery, Bakery & Snacks (製菓・スナック・製パン) vs 1617.T: rho = 0.4746
- Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材) vs 1617.T: rho = 0.4746
- Cosmetics & Personal Care (化粧品・トイレタリー・日用品) vs 1620.T: rho = 0.3647
- Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品) vs 1626.T: rho = 0.3639
- Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設) vs 1630.T: rho = 0.2983
- Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア) vs 1630.T: rho = 0.4482
- Drugstores & Dispensing (ドラッグストア・調剤薬局) vs 1630.T: rho = 0.3588
- Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール) vs 1630.T: rho = 0.2748
- Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売) vs 1630.T: rho = 0.4203
- Restaurants & Food Service (外食・フードサービス) vs 1630.T: rho = 0.3298
- General Trading Companies - Sogo Shosha (総合商社) vs 1629.T: rho = 0.6941
- Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材) vs 1629.T: rho = 0.5339
- Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸) vs 1629.T: rho = 0.3413
- Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀) vs 1631.T: rho = 0.6904
- Regional & Provincial Banking Groups (地域ブロック中核地銀) vs 1631.T: rho = 0.6777
- Digital, Retail & Specialized Banks (ネット・流通・専門銀行) vs 1631.T: rho = 0.5912
- Securities, Exchanges & Investment (証券・取引所・投資育成) vs 1632.T: rho = 0.5528
- Life & Non-Life Insurance (生命保険・損害保険) vs 1632.T: rho = 0.6240
- Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融) vs 1632.T: rho = 0.5539
- Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融) vs 1632.T: rho = 0.4779
- Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発) vs 1628.T: rho = 0.5759
- Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通) vs 1628.T: rho = 0.5436
- JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網) vs 1628.T: rho = 0.5876
- Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング) vs 1628.T: rho = 0.4002
- Marine Shipping (外航海運・海上輸送) vs 1628.T: rho = 0.3338
- Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流) vs 1628.T: rho = 0.3852
- Airlines & Airport Infrastructure (航空・空港インフラ) vs 1628.T: rho = 0.4234
- City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給) vs 1627.T: rho = 0.5832
- Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発) vs 1618.T: rho = 0.6863
- Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通) vs 1626.T: rho = 0.3483
- Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター) vs 1626.T: rho = 0.4544
- Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS) vs 1626.T: rho = 0.4021
- Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア) vs 1626.T: rho = 0.4032
- Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網) vs 1626.T: rho = 0.3450
- IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通) vs 1626.T: rho = 0.4212
- Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援) vs 1626.T: rho = 0.3031
- HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生) vs 1626.T: rho = 0.3761
- Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援) vs 1626.T: rho = 0.4039
- Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー) vs 1626.T: rho = 0.4262
- Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス) vs 1626.T: rho = 0.4030
- Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント) vs 1624.T: rho = 0.2653
- Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送) vs 1626.T: rho = 0.3873
- Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー) vs 1626.T: rho = 0.3582

## Sector coverage
| sector_etf | covered | total_jpx | coverage_count |
|---|---|---|---|
| 1617.T | 29 | 138 | 0.21014492753623187 |
| 1618.T | 4 | 14 | 0.2857142857142857 |
| 1619.T | 38 | 294 | 0.1292517006802721 |
| 1620.T | 48 | 275 | 0.17454545454545456 |
| 1621.T | 16 | 80 | 0.2 |
| 1622.T | 21 | 98 | 0.21428571428571427 |
| 1623.T | 14 | 70 | 0.2 |
| 1624.T | 33 | 211 | 0.15639810426540285 |
| 1625.T | 60 | 280 | 0.21428571428571427 |
| 1626.T | 68 | 1328 | 0.05120481927710843 |
| 1627.T | 13 | 29 | 0.4482758620689655 |
| 1628.T | 34 | 109 | 0.3119266055045872 |
| 1629.T | 25 | 301 | 0.08305647840531562 |
| 1630.T | 33 | 342 | 0.09649122807017543 |
| 1631.T | 26 | 79 | 0.3291139240506329 |
| 1632.T | 20 | 96 | 0.20833333333333334 |
| 1633.T | 12 | 159 | 0.07547169811320754 |


## All subsector vs dominant ETF correlations
| subsector | etf | rho |
|---|---|---|
| General Contractors (ゼネコン・総合建設) | 1619.T | 0.5681106284436646 |
| Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材) | 1619.T | 0.574226600594432 |
| Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事) | 1619.T | 0.5445813944538077 |
| Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ) | 1619.T | 0.48643023947328556 |
| Comprehensive Real Estate Developers (総合不動産・ビルデベロッパー) | 1633.T | 0.7755419730800693 |
| Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材) | 1620.T | 0.4756685635610476 |
| Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル) | 1620.T | 0.5143504766952753 |
| Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材) | 1620.T | 0.502221777278076 |
| Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料) | 1620.T | 0.40819339626004353 |
| Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル) | 1620.T | 0.47516949559013427 |
| Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材) | 1619.T | 0.5275544758586982 |
| Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷) | 1620.T | 0.38723790385128404 |
| Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板) | 1623.T | 0.5873762997069827 |
| Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料) | 1623.T | 0.5958625978614353 |
| Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ) | 1623.T | 0.6129036900092557 |
| Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装) | 1624.T | 0.5122409672983923 |
| Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器) | 1624.T | 0.478154115116939 |
| Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械) | 1624.T | 0.5585735205865591 |
| Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング) | 1624.T | 0.5471204874947587 |
| Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池) | 1625.T | 0.5716370737401651 |
| Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE) | 1625.T | 0.5491685358998442 |
| Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品) | 1625.T | 0.6036442596585587 |
| Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ) | 1625.T | 0.5370732756297794 |
| Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学) | 1625.T | 0.41286888007608114 |
| Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器) | 1625.T | 0.5291715300006743 |
| Passenger Cars, Commercial Vehicles & Motorcycles (乗用車・完成車・商用トラック・二輪) | 1622.T | 0.752449353151117 |
| Auto Parts & Systems (自動車部品・電装品・内装) | 1622.T | 0.6729731650171376 |
| Tires & Rubber Products (タイヤ・ゴム製品) | 1622.T | 0.580282388542947 |
| Global Major Innovative Pharma (グローバル新薬・大型創薬) | 1621.T | 0.7433071832196224 |
| Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック) | 1621.T | 0.5291325485668976 |
| Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品) | 1621.T | 0.5286568833263235 |
| Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ) | 1617.T | 0.5644226660235422 |
| Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品) | 1617.T | 0.46429744092405945 |
| Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品) | 1617.T | 0.5513157443939203 |
| Confectionery, Bakery & Snacks (製菓・スナック・製パン) | 1617.T | 0.4745591333924797 |
| Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材) | 1617.T | 0.47458678793257003 |
| Cosmetics & Personal Care (化粧品・トイレタリー・日用品) | 1620.T | 0.36470573262831674 |
| Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品) | 1626.T | 0.3638981156348085 |
| Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設) | 1630.T | 0.2983156541415584 |
| Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア) | 1630.T | 0.44817733881169697 |
| Drugstores & Dispensing (ドラッグストア・調剤薬局) | 1630.T | 0.35877349026683847 |
| Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール) | 1630.T | 0.2748387067659832 |
| Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売) | 1630.T | 0.420345092033393 |
| Restaurants & Food Service (外食・フードサービス) | 1630.T | 0.3297794786203562 |
| General Trading Companies - Sogo Shosha (総合商社) | 1629.T | 0.6940883915243797 |
| Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材) | 1629.T | 0.5338848799025862 |
| Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸) | 1629.T | 0.341254164629362 |
| Megabanks & Trust Banks (メガバンク・ゆうちょ・信託) | 1631.T | 0.8106915706337414 |
| Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀) | 1631.T | 0.6904413540850282 |
| Regional & Provincial Banking Groups (地域ブロック中核地銀) | 1631.T | 0.6777131610453453 |
| Digital, Retail & Specialized Banks (ネット・流通・専門銀行) | 1631.T | 0.5912433125100471 |
| Securities, Exchanges & Investment (証券・取引所・投資育成) | 1632.T | 0.5528487427696724 |
| Life & Non-Life Insurance (生命保険・損害保険) | 1632.T | 0.6239660204873625 |
| Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融) | 1632.T | 0.5538572574098889 |
| Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融) | 1632.T | 0.4778784380270634 |
| Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発) | 1628.T | 0.5759092790784068 |
| Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通) | 1628.T | 0.5435755724023219 |
| JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網) | 1628.T | 0.587559106325646 |
| Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング) | 1628.T | 0.4002429893391228 |
| Marine Shipping (外航海運・海上輸送) | 1628.T | 0.3337582689147425 |
| Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流) | 1628.T | 0.385181859628668 |
| Airlines & Airport Infrastructure (航空・空港インフラ) | 1628.T | 0.4234352186662692 |
| Electric Power Utilities (電力・発電・送配電) | 1627.T | 0.823114532123357 |
| City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給) | 1627.T | 0.5831518591652586 |
| Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発) | 1618.T | 0.6862664276522107 |
| Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通) | 1626.T | 0.34826762168613057 |
| Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター) | 1626.T | 0.4544124683390079 |
| Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS) | 1626.T | 0.4020990486870188 |
| Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア) | 1626.T | 0.40324148679777116 |
| Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網) | 1626.T | 0.34504839219113675 |
| IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通) | 1626.T | 0.42117196886043884 |
| Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援) | 1626.T | 0.3031026380579285 |
| HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生) | 1626.T | 0.3761084480149556 |
| Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援) | 1626.T | 0.4038697940455013 |
| Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー) | 1626.T | 0.4262026175138387 |
| Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス) | 1626.T | 0.40303440159932474 |
| Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント) | 1624.T | 0.2652754302946393 |
| Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送) | 1626.T | 0.3873372386485117 |
| Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー) | 1626.T | 0.35823920509029095 |

