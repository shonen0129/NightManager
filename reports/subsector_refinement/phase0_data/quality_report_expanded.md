# Phase 0: Subsector Panel Quality Report
**Date**: 2026-08-23
**Panel files**: /Users/shonen/leadlag/var/research/subsector/panel_subsector_oc_expanded.parquet
**Rows (dates)**: 4168
**Subsectors**: 79
**Overall valid observation ratio**: 0.9442

## Phase 0 Gates
- (1) Unmapped tickers in canonical mapping: 0 (need 0) — see excluded_unmapped section
- (2) All 17 sectors covered: True (need True)
- (3) Basket vs ETF correlation >= 0.7: 76 subsectors below threshold; min rho = 0.3711; median = 0.5293 (need min >= 0.7)
- (4) Sector coverage measured and recorded: True
- (5) Baseline (2010-2014) subsectors with >=126 valid days: 79 (need >= 60)

## Aggregation matrix A validation
- Row sums close to 1.0 or empty: True
- Empty rows (non-covered sectors): 0
- Shape: (17, 79)

## Subsectors with rho < 0.7
- General Contractors (ゼネコン・総合建設) vs 1619.T: rho = 0.5890
- Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材) vs 1619.T: rho = 0.6075
- Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事) vs 1619.T: rho = 0.5805
- Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ) vs 1619.T: rho = 0.5678
- Comprehensive Real Estate Developers (総合不動産・ビルデベロッパー) vs 1633.T: rho = 0.6468
- Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材) vs 1620.T: rho = 0.5066
- Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル) vs 1620.T: rho = 0.5188
- Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材) vs 1620.T: rho = 0.5088
- Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料) vs 1620.T: rho = 0.4898
- Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル) vs 1620.T: rho = 0.5081
- Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材) vs 1619.T: rho = 0.5718
- Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷) vs 1620.T: rho = 0.4019
- Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板) vs 1623.T: rho = 0.5757
- Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料) vs 1623.T: rho = 0.6003
- Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ) vs 1623.T: rho = 0.6360
- Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装) vs 1624.T: rho = 0.5333
- Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器) vs 1624.T: rho = 0.5109
- Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械) vs 1624.T: rho = 0.5599
- Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング) vs 1624.T: rho = 0.5294
- Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池) vs 1625.T: rho = 0.5743
- Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE) vs 1625.T: rho = 0.5875
- Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品) vs 1625.T: rho = 0.5999
- Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ) vs 1625.T: rho = 0.5619
- Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学) vs 1625.T: rho = 0.5000
- Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器) vs 1625.T: rho = 0.5712
- Auto Parts & Systems (自動車部品・電装品・内装) vs 1622.T: rho = 0.6713
- Tires & Rubber Products (タイヤ・ゴム製品) vs 1622.T: rho = 0.6013
- Global Major Innovative Pharma (グローバル新薬・大型創薬) vs 1621.T: rho = 0.6988
- Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック) vs 1621.T: rho = 0.5888
- Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品) vs 1621.T: rho = 0.5498
- Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ) vs 1617.T: rho = 0.5483
- Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品) vs 1617.T: rho = 0.5081
- Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品) vs 1617.T: rho = 0.5574
- Confectionery, Bakery & Snacks (製菓・スナック・製パン) vs 1617.T: rho = 0.5031
- Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材) vs 1617.T: rho = 0.4889
- Cosmetics & Personal Care (化粧品・トイレタリー・日用品) vs 1620.T: rho = 0.4970
- Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品) vs 1626.T: rho = 0.4098
- Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設) vs 1630.T: rho = 0.3959
- Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア) vs 1630.T: rho = 0.4199
- Drugstores & Dispensing (ドラッグストア・調剤薬局) vs 1630.T: rho = 0.4036
- Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール) vs 1630.T: rho = 0.3811
- Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売) vs 1630.T: rho = 0.4090
- Restaurants & Food Service (外食・フードサービス) vs 1630.T: rho = 0.3977
- General Trading Companies - Sogo Shosha (総合商社) vs 1629.T: rho = 0.6184
- Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材) vs 1629.T: rho = 0.5498
- Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸) vs 1629.T: rho = 0.5345
- Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀) vs 1631.T: rho = 0.6765
- Regional & Provincial Banking Groups (地域ブロック中核地銀) vs 1631.T: rho = 0.6797
- Digital, Retail & Specialized Banks (ネット・流通・専門銀行) vs 1631.T: rho = 0.6778
- Securities, Exchanges & Investment (証券・取引所・投資育成) vs 1632.T: rho = 0.5485
- Life & Non-Life Insurance (生命保険・損害保険) vs 1632.T: rho = 0.5958
- Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融) vs 1632.T: rho = 0.5223
- Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融) vs 1632.T: rho = 0.5255
- Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発) vs 1628.T: rho = 0.5805
- Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通) vs 1628.T: rho = 0.5293
- JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網) vs 1628.T: rho = 0.5618
- Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング) vs 1628.T: rho = 0.4036
- Marine Shipping (外航海運・海上輸送) vs 1628.T: rho = 0.3711
- Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流) vs 1628.T: rho = 0.4233
- Airlines & Airport Infrastructure (航空・空港インフラ) vs 1628.T: rho = 0.3859
- City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給) vs 1627.T: rho = 0.5394
- Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発) vs 1618.T: rho = 0.6406
- Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通) vs 1626.T: rho = 0.4177
- Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター) vs 1626.T: rho = 0.4590
- Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS) vs 1626.T: rho = 0.4330
- Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア) vs 1626.T: rho = 0.4465
- Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網) vs 1626.T: rho = 0.4276
- IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通) vs 1626.T: rho = 0.4489
- Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援) vs 1626.T: rho = 0.4303
- HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生) vs 1626.T: rho = 0.4277
- Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援) vs 1626.T: rho = 0.4292
- Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー) vs 1626.T: rho = 0.4612
- Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス) vs 1626.T: rho = 0.4481
- Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント) vs 1624.T: rho = 0.4999
- Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送) vs 1626.T: rho = 0.4570
- Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー) vs 1626.T: rho = 0.4296

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
| General Contractors (ゼネコン・総合建設) | 1619.T | 0.5889648269670149 |
| Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材) | 1619.T | 0.6075193671194475 |
| Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事) | 1619.T | 0.580527669156556 |
| Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ) | 1619.T | 0.5677670077683408 |
| Comprehensive Real Estate Developers (総合不動産・ビルデベロッパー) | 1633.T | 0.6468481646265785 |
| Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材) | 1620.T | 0.5065767041725746 |
| Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル) | 1620.T | 0.5187895782587174 |
| Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材) | 1620.T | 0.5087741763335125 |
| Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料) | 1620.T | 0.4898056680863022 |
| Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル) | 1620.T | 0.5081300891629761 |
| Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材) | 1619.T | 0.5718405860991757 |
| Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷) | 1620.T | 0.40187898896234103 |
| Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板) | 1623.T | 0.5756987212637877 |
| Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料) | 1623.T | 0.600262196898343 |
| Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ) | 1623.T | 0.6359807674392586 |
| Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装) | 1624.T | 0.5333499316413415 |
| Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器) | 1624.T | 0.5108796877851763 |
| Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械) | 1624.T | 0.5599438771244443 |
| Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング) | 1624.T | 0.5293672258048646 |
| Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池) | 1625.T | 0.5742543349449624 |
| Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE) | 1625.T | 0.5874610062943912 |
| Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品) | 1625.T | 0.5998996945582746 |
| Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ) | 1625.T | 0.5618893269548201 |
| Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学) | 1625.T | 0.49996196406184973 |
| Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器) | 1625.T | 0.5711643243039942 |
| Passenger Cars, Commercial Vehicles & Motorcycles (乗用車・完成車・商用トラック・二輪) | 1622.T | 0.7105827006134454 |
| Auto Parts & Systems (自動車部品・電装品・内装) | 1622.T | 0.6713274803406596 |
| Tires & Rubber Products (タイヤ・ゴム製品) | 1622.T | 0.6012846904509499 |
| Global Major Innovative Pharma (グローバル新薬・大型創薬) | 1621.T | 0.6988336901200058 |
| Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック) | 1621.T | 0.5888174885212455 |
| Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品) | 1621.T | 0.5498041132083519 |
| Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ) | 1617.T | 0.5483393582851621 |
| Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品) | 1617.T | 0.5080687576688848 |
| Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品) | 1617.T | 0.5574291055211591 |
| Confectionery, Bakery & Snacks (製菓・スナック・製パン) | 1617.T | 0.5030666350605397 |
| Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材) | 1617.T | 0.48885922185891856 |
| Cosmetics & Personal Care (化粧品・トイレタリー・日用品) | 1620.T | 0.4970335438095092 |
| Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品) | 1626.T | 0.4097672261205761 |
| Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設) | 1630.T | 0.39591492941818396 |
| Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア) | 1630.T | 0.41987921527170946 |
| Drugstores & Dispensing (ドラッグストア・調剤薬局) | 1630.T | 0.4036253652119076 |
| Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール) | 1630.T | 0.38114550960674265 |
| Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売) | 1630.T | 0.40898309887901996 |
| Restaurants & Food Service (外食・フードサービス) | 1630.T | 0.3977028455346979 |
| General Trading Companies - Sogo Shosha (総合商社) | 1629.T | 0.6184114672502015 |
| Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材) | 1629.T | 0.5498191931337707 |
| Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸) | 1629.T | 0.5345189772178179 |
| Megabanks & Trust Banks (メガバンク・ゆうちょ・信託) | 1631.T | 0.7503754021009539 |
| Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀) | 1631.T | 0.6764564543687523 |
| Regional & Provincial Banking Groups (地域ブロック中核地銀) | 1631.T | 0.6796662701803783 |
| Digital, Retail & Specialized Banks (ネット・流通・専門銀行) | 1631.T | 0.6777745088251614 |
| Securities, Exchanges & Investment (証券・取引所・投資育成) | 1632.T | 0.5484997714741973 |
| Life & Non-Life Insurance (生命保険・損害保険) | 1632.T | 0.595789835922549 |
| Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融) | 1632.T | 0.5222898136700036 |
| Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融) | 1632.T | 0.5255288564943785 |
| Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発) | 1628.T | 0.5804978680103643 |
| Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通) | 1628.T | 0.5293278316272966 |
| JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網) | 1628.T | 0.5618050206831592 |
| Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング) | 1628.T | 0.403631609667314 |
| Marine Shipping (外航海運・海上輸送) | 1628.T | 0.37107925817316745 |
| Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流) | 1628.T | 0.4233001281371682 |
| Airlines & Airport Infrastructure (航空・空港インフラ) | 1628.T | 0.3858919738733966 |
| Electric Power Utilities (電力・発電・送配電) | 1627.T | 0.82036473432538 |
| City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給) | 1627.T | 0.539358348818522 |
| Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発) | 1618.T | 0.6405575253731203 |
| Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通) | 1626.T | 0.4176732771336527 |
| Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター) | 1626.T | 0.4589887985814109 |
| Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS) | 1626.T | 0.43295423183524345 |
| Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア) | 1626.T | 0.4465332881541968 |
| Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網) | 1626.T | 0.4275534183291599 |
| IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通) | 1626.T | 0.448924649881822 |
| Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援) | 1626.T | 0.4303300409020443 |
| HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生) | 1626.T | 0.4276824740357952 |
| Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援) | 1626.T | 0.4292284531632303 |
| Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー) | 1626.T | 0.46115941458092324 |
| Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス) | 1626.T | 0.4480856499322311 |
| Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント) | 1624.T | 0.4999188820692799 |
| Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送) | 1626.T | 0.45696173739223855 |
| Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー) | 1626.T | 0.42959131323523625 |

