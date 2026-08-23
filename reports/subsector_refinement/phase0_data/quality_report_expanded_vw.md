# Phase 0: Subsector Panel Quality Report
**Date**: 2026-08-23
**Panel files**: /Users/shonen/leadlag/var/research/subsector/panel_subsector_oc_expanded_vw.parquet
**Rows (dates)**: 4168
**Subsectors**: 79
**Overall valid observation ratio**: 0.9442

## Phase 0 Gates
- (1) Unmapped tickers in canonical mapping: 0 (need 0) — see excluded_unmapped section
- (2) All 17 sectors covered: True (need True)
- (3) Basket vs ETF correlation >= 0.7: 74 subsectors below threshold; min rho = 0.3556; median = 0.5395 (need min >= 0.7)
- (4) Sector coverage measured and recorded: True
- (5) Baseline (2010-2014) subsectors with >=126 valid days: 79 (need >= 60)

## Aggregation matrix A validation
- Row sums close to 1.0 or empty: True
- Empty rows (non-covered sectors): 0
- Shape: (17, 79)

## Subsectors with rho < 0.7
- General Contractors (ゼネコン・総合建設) vs 1619.T: rho = 0.5927
- Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材) vs 1619.T: rho = 0.6215
- Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事) vs 1619.T: rho = 0.5802
- Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ) vs 1619.T: rho = 0.5808
- Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材) vs 1620.T: rho = 0.5027
- Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル) vs 1620.T: rho = 0.5324
- Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材) vs 1620.T: rho = 0.5204
- Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料) vs 1620.T: rho = 0.4900
- Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル) vs 1620.T: rho = 0.5143
- Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材) vs 1619.T: rho = 0.5676
- Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷) vs 1620.T: rho = 0.3981
- Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板) vs 1623.T: rho = 0.5940
- Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料) vs 1623.T: rho = 0.6123
- Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ) vs 1623.T: rho = 0.6529
- Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装) vs 1624.T: rho = 0.5395
- Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器) vs 1624.T: rho = 0.5186
- Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械) vs 1624.T: rho = 0.5782
- Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング) vs 1624.T: rho = 0.5509
- Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池) vs 1625.T: rho = 0.5966
- Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE) vs 1625.T: rho = 0.5721
- Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品) vs 1625.T: rho = 0.6179
- Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ) vs 1625.T: rho = 0.5709
- Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学) vs 1625.T: rho = 0.4897
- Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器) vs 1625.T: rho = 0.5787
- Auto Parts & Systems (自動車部品・電装品・内装) vs 1622.T: rho = 0.6810
- Tires & Rubber Products (タイヤ・ゴム製品) vs 1622.T: rho = 0.6147
- Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック) vs 1621.T: rho = 0.5847
- Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品) vs 1621.T: rho = 0.5660
- Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ) vs 1617.T: rho = 0.5744
- Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品) vs 1617.T: rho = 0.5223
- Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品) vs 1617.T: rho = 0.5610
- Confectionery, Bakery & Snacks (製菓・スナック・製パン) vs 1617.T: rho = 0.5154
- Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材) vs 1617.T: rho = 0.4947
- Cosmetics & Personal Care (化粧品・トイレタリー・日用品) vs 1620.T: rho = 0.4569
- Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品) vs 1626.T: rho = 0.4092
- Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設) vs 1630.T: rho = 0.3903
- Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア) vs 1630.T: rho = 0.4650
- Drugstores & Dispensing (ドラッグストア・調剤薬局) vs 1630.T: rho = 0.4057
- Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール) vs 1630.T: rho = 0.3751
- Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売) vs 1630.T: rho = 0.4313
- Restaurants & Food Service (外食・フードサービス) vs 1630.T: rho = 0.3944
- General Trading Companies - Sogo Shosha (総合商社) vs 1629.T: rho = 0.6950
- Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材) vs 1629.T: rho = 0.5548
- Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸) vs 1629.T: rho = 0.5026
- Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀) vs 1631.T: rho = 0.6985
- Regional & Provincial Banking Groups (地域ブロック中核地銀) vs 1631.T: rho = 0.6831
- Digital, Retail & Specialized Banks (ネット・流通・専門銀行) vs 1631.T: rho = 0.6906
- Securities, Exchanges & Investment (証券・取引所・投資育成) vs 1632.T: rho = 0.5710
- Life & Non-Life Insurance (生命保険・損害保険) vs 1632.T: rho = 0.6287
- Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融) vs 1632.T: rho = 0.5579
- Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融) vs 1632.T: rho = 0.5178
- Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発) vs 1628.T: rho = 0.5779
- Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通) vs 1628.T: rho = 0.5441
- JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網) vs 1628.T: rho = 0.6023
- Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング) vs 1628.T: rho = 0.4064
- Marine Shipping (外航海運・海上輸送) vs 1628.T: rho = 0.3556
- Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流) vs 1628.T: rho = 0.4243
- Airlines & Airport Infrastructure (航空・空港インフラ) vs 1628.T: rho = 0.4234
- City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給) vs 1627.T: rho = 0.6039
- Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発) vs 1618.T: rho = 0.6901
- Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通) vs 1626.T: rho = 0.4650
- Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター) vs 1626.T: rho = 0.5182
- Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS) vs 1626.T: rho = 0.4489
- Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア) vs 1626.T: rho = 0.4506
- Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網) vs 1626.T: rho = 0.4228
- IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通) vs 1626.T: rho = 0.4528
- Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援) vs 1626.T: rho = 0.4121
- HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生) vs 1626.T: rho = 0.4458
- Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援) vs 1626.T: rho = 0.4527
- Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー) vs 1626.T: rho = 0.4693
- Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス) vs 1626.T: rho = 0.4584
- Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント) vs 1624.T: rho = 0.4942
- Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送) vs 1626.T: rho = 0.4704
- Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー) vs 1626.T: rho = 0.4355

## Sector coverage
| coverage_count | covered | sector_etf | total_jpx |
|---|---|---|---|
| 0.036231884057971016 | 5 | 1617.T | 138 |
| 0.07142857142857142 | 1 | 1618.T | 14 |
| 0.027210884353741496 | 8 | 1619.T | 294 |
| 0.03636363636363636 | 10 | 1620.T | 275 |
| 0.0375 | 3 | 1621.T | 80 |
| 0.05102040816326531 | 5 | 1622.T | 98 |
| 0.04285714285714286 | 3 | 1623.T | 70 |
| 0.02843601895734597 | 6 | 1624.T | 211 |
| 0.03571428571428571 | 10 | 1625.T | 280 |
| 0.01355421686746988 | 18 | 1626.T | 1328 |
| 0.06896551724137931 | 2 | 1627.T | 29 |
| 0.06422018348623854 | 7 | 1628.T | 109 |
| 0.019933554817275746 | 6 | 1629.T | 301 |
| 0.02631578947368421 | 9 | 1630.T | 342 |
| 0.05063291139240506 | 4 | 1631.T | 79 |
| 0.052083333333333336 | 5 | 1632.T | 96 |
| 0.025157232704402517 | 4 | 1633.T | 159 |


## All subsector vs dominant ETF correlations
| subsector | etf | rho |
|---|---|---|
| General Contractors (ゼネコン・総合建設) | 1619.T | 0.5926549575804021 |
| Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材) | 1619.T | 0.6215416220949167 |
| Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事) | 1619.T | 0.5802278372934092 |
| Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ) | 1619.T | 0.5807860284185618 |
| Comprehensive Real Estate Developers (総合不動産・ビルデベロッパー) | 1633.T | 0.7443927215861521 |
| Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材) | 1620.T | 0.5027410353256337 |
| Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル) | 1620.T | 0.5323853301181951 |
| Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材) | 1620.T | 0.5204056181483584 |
| Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料) | 1620.T | 0.4900007753127009 |
| Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル) | 1620.T | 0.5143470866962438 |
| Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材) | 1619.T | 0.5676350410043874 |
| Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷) | 1620.T | 0.3981400784294305 |
| Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板) | 1623.T | 0.5939811802615568 |
| Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料) | 1623.T | 0.6123212730468403 |
| Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ) | 1623.T | 0.6528970468875513 |
| Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装) | 1624.T | 0.5395071030906906 |
| Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器) | 1624.T | 0.5186113595051692 |
| Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械) | 1624.T | 0.578171790575645 |
| Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング) | 1624.T | 0.5509414305002445 |
| Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池) | 1625.T | 0.5966438181263142 |
| Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE) | 1625.T | 0.572088024111101 |
| Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品) | 1625.T | 0.6178953622580737 |
| Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ) | 1625.T | 0.5709175738612468 |
| Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学) | 1625.T | 0.4896545327330681 |
| Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器) | 1625.T | 0.5787004239441419 |
| Passenger Cars, Commercial Vehicles & Motorcycles (乗用車・完成車・商用トラック・二輪) | 1622.T | 0.7549795786428489 |
| Auto Parts & Systems (自動車部品・電装品・内装) | 1622.T | 0.680963363426817 |
| Tires & Rubber Products (タイヤ・ゴム製品) | 1622.T | 0.6147153325682875 |
| Global Major Innovative Pharma (グローバル新薬・大型創薬) | 1621.T | 0.7434800386949186 |
| Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック) | 1621.T | 0.584678481920197 |
| Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品) | 1621.T | 0.5659935629837342 |
| Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ) | 1617.T | 0.5744185527623108 |
| Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品) | 1617.T | 0.5223314357065119 |
| Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品) | 1617.T | 0.5609968623460306 |
| Confectionery, Bakery & Snacks (製菓・スナック・製パン) | 1617.T | 0.5154435596076281 |
| Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材) | 1617.T | 0.49471728369673995 |
| Cosmetics & Personal Care (化粧品・トイレタリー・日用品) | 1620.T | 0.4568530667036974 |
| Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品) | 1626.T | 0.40918771083713473 |
| Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設) | 1630.T | 0.3903036443096068 |
| Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア) | 1630.T | 0.46501270566401176 |
| Drugstores & Dispensing (ドラッグストア・調剤薬局) | 1630.T | 0.40569129345492433 |
| Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール) | 1630.T | 0.3750871789424258 |
| Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売) | 1630.T | 0.4313420416950202 |
| Restaurants & Food Service (外食・フードサービス) | 1630.T | 0.3944357163589334 |
| General Trading Companies - Sogo Shosha (総合商社) | 1629.T | 0.6950491289906383 |
| Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材) | 1629.T | 0.5547971901637195 |
| Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸) | 1629.T | 0.5026287423726369 |
| Megabanks & Trust Banks (メガバンク・ゆうちょ・信託) | 1631.T | 0.8143950059853271 |
| Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀) | 1631.T | 0.6985468071123873 |
| Regional & Provincial Banking Groups (地域ブロック中核地銀) | 1631.T | 0.6831116461986867 |
| Digital, Retail & Specialized Banks (ネット・流通・専門銀行) | 1631.T | 0.6906055012943523 |
| Securities, Exchanges & Investment (証券・取引所・投資育成) | 1632.T | 0.5709839914435842 |
| Life & Non-Life Insurance (生命保険・損害保険) | 1632.T | 0.6286739930232338 |
| Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融) | 1632.T | 0.5579153940634279 |
| Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融) | 1632.T | 0.517790524109302 |
| Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発) | 1628.T | 0.5779279131502151 |
| Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通) | 1628.T | 0.5440702557145078 |
| JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網) | 1628.T | 0.6022907655452552 |
| Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング) | 1628.T | 0.4064329101891549 |
| Marine Shipping (外航海運・海上輸送) | 1628.T | 0.35564761339583495 |
| Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流) | 1628.T | 0.424314993141894 |
| Airlines & Airport Infrastructure (航空・空港インフラ) | 1628.T | 0.4234352186662692 |
| Electric Power Utilities (電力・発電・送配電) | 1627.T | 0.8254667201746893 |
| City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給) | 1627.T | 0.6039069761690873 |
| Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発) | 1618.T | 0.690089235800627 |
| Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通) | 1626.T | 0.4649880411230161 |
| Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター) | 1626.T | 0.5181525501911879 |
| Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS) | 1626.T | 0.44892205229073817 |
| Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア) | 1626.T | 0.45057349382691636 |
| Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網) | 1626.T | 0.4228404269329975 |
| IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通) | 1626.T | 0.4527686680209819 |
| Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援) | 1626.T | 0.41207944143521585 |
| HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生) | 1626.T | 0.4458216809665133 |
| Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援) | 1626.T | 0.45271609716330885 |
| Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー) | 1626.T | 0.4692760844668446 |
| Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス) | 1626.T | 0.4583723261103937 |
| Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント) | 1624.T | 0.4942198495171635 |
| Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送) | 1626.T | 0.4704052707075955 |
| Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー) | 1626.T | 0.43549883378671744 |

