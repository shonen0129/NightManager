# Phase 0: Subsector Mapping Validation
**Date**: 2026-08-23
**Taxonomy**: taxonomy_subsectors.yaml (79 subsectors, 498 tickers)
**JPX master**: data_j.xls
**Unmapped tickers**: 4 / 498

## Aggregation Matrix A (17 x 79)
Rows: TOPIX-17 ETF, Cols: subsector. Values are equal-weight stock counts / sector total.

## Unmapped tickers (missing in JPX master)
- 6406.T (Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング))
- 6201.T (Auto Parts & Systems (自動車部品・電装品・内装))
- 4530.T (Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品))
- 9719.T (IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通))

## Cross-tabulation (subsector x TOPIX-17 ETF)
| index | subsector | 1617.T | 1618.T | 1619.T | 1620.T | 1621.T | 1622.T | 1623.T | 1624.T | 1625.T | 1626.T | 1627.T | 1628.T | 1629.T | 1630.T | 1631.T | 1632.T | 1633.T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材) | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | Advertising Agencies & Commercial Broadcasting (広告代理店・民放テレビ放送) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | Airlines & Airport Infrastructure (航空・空港インフラ) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 |
| 3 | Apparel, Footwear, Sports Brands & Lifestyle SPA (アパレル・フットウェア・スポーツブランド・生活製造小売) | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 0 | 0 | 0 |
| 4 | Auto Parts & Systems (自動車部品・電装品・内装) | 0 | 0 | 1 | 1 | 0 | 6 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | Basic & Diversified Chemicals (総合・基礎化学・エチレン・機能素材) | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | Beverages, Breweries & Distilling (飲料・ビール・酒類・たばこ) | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | Character Toys, Anime, Film & Creative IP (キャラクター玩具・アニメ・映画・IPライセンス) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 8 | City Gas, Industrial Gas & Energy Utilities (都市ガス・LPガス・産業ガス・エネルギー供給) | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 0 |
| 9 | Comprehensive Real Estate Developers (総合不動産・ビルデベロッパー) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| 10 | Confectionery, Bakery & Snacks (製菓・スナック・製パン) | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 | Construction, Mining, Defense & Heavy Machinery (建設機械・鉱山機械・防衛重工・農業機械) | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | Consulting, M&A & DX Quality Services (コンサルティング・M&A・DX品質支援) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | Consumer Credit, Cards & Retail Finance (信販・クレジットカード・消費者金融) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 4 | 0 |
| 14 | Consumer Electronics Mass Retailers (家電量販店・生活家電・免税リテール) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| 15 | Consumer Healthcare, OTC, Kampo & Generics (OTC医薬品・漢方・スキンケア・後発品) | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | Consumer Internet, Portals & E-Commerce (ネットポータル・ECモール・Webメディア) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| 17 | Corporate Leasing & Capital Asset Finance (総合リース・法人設備金融) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 18 | Cosmetics & Personal Care (化粧品・トイレタリー・日用品) | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 19 | Dairy, Probiotics & Nutritional Foods (乳業・プロバイオティクス・栄養食品) | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | Department Stores & Luxury Commercial Real Estate (百貨店・高級ブランド・都市型商業施設) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| 21 | Digital, Retail & Specialized Banks (ネット・流通・専門銀行) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| 22 | Drugstores & Dispensing (ドラッグストア・調剤薬局) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 |
| 23 | Electric Power Utilities (電力・発電・送配電) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 | Electronic Components, Sensors & Passive Devices (電子部品・センサ・受動部品) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 25 | Energy, Petroleum Refining & Resources (エネルギー・石油元売・資源開発) | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 26 | Enterprise Software, ERP & Cloud SaaS (業務ソフトウェア・ERP・クラウドSaaS) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 27 | General Contractors (ゼネコン・総合建設) | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 28 | General Trading Companies - Sogo Shosha (総合商社) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| 29 | Glass, Advanced Ceramics & Cement (ガラス・先端セラミックス・セメント・断熱材) | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30 | Global Gaming Platforms & Digital Publishers (グローバルゲーム・デジタルパブリッシャー) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 31 | Global Major Innovative Pharma (グローバル新薬・大型創薬) | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | Grain Milling, Edible Oils, Marine & Agribusiness (製粉・製油・水産・アグリ素材) | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 33 | Greater Tokyo Urban Transit & TOD (首都圏大手私鉄・都市交通・沿線再開発) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 |
| 34 | HR, Staffing & Relocation Services (人材・HR・総合派遣・社宅福利厚生) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 35 | Heavy Electrical, Power Systems & Batteries (重電・電力グリッド・産業パワー・電池) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 36 | Housing Fixtures, Sanitary Ware & Building Openings (住宅設備・衛生陶器・建材・サッシ) | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 37 | Housing, Homebuilders & Wood Products (住宅・ハウスメーカー・木材) | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| 38 | IT Services & Enterprise SIers (大型SIer・システム受託開発・IT流通) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 39 | Industrial Automation, Robotics & Process Control (FA・産業ロボット・プロセス制御計装) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 40 | Industrial Machinery, HVAC & Precision Bearings (産業機械・空調設備・油圧・精密ベアリング) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 41 | Integrated Steel Mills & Specialty Steel (高炉・特殊鋼・電炉・鋼板) | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 42 | Internet Infrastructure & FinTech Payments (ネットインフラ・決済代行・通信網) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43 | JR Group & Intercity High-Speed Rail (JR旅客鉄道グループ・新幹線幹線網) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| 44 | Kansai & Inter-Regional Private Railways (関西・広域大手私鉄・観光交通) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| 45 | Life & Non-Life Insurance (生命保険・損害保険) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| 46 | Logistics, Trucking & Forwarding (トラック・総合物流・フォワーディング) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 0 |
| 47 | Machine Tools, Cutting & Commercial Automation (工作機械・切削工具・板金・自動化機器) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 48 | Major Metropolitan & Super-Regional Banks (大都市圏・広域メガ地銀) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| 49 | Marine Shipping (外航海運・海上輸送) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| 50 | Medical Devices, Diagnostics & Clinical Testing (医療機器・臨床検査・生体医工学) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 51 | Megabanks & Trust Banks (メガバンク・ゆうちょ・信託) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| 52 | Non-Ferrous Smelting, Mining & Advanced Materials (非鉄製錬・資源鉱山・先端金属材料) | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 53 | Office Printing, Consumer Electronics & Industrial Devices (オフィスOA・プリンティング・民生機器) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 54 | Optical Fiber, Power Cables & Communication Infrastructure (電線・光ファイバ・電力通信インフラ) | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 55 | Optical Instruments, Analytical & Precision Measurement (精密光学機器・分析計測・カメラ) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 56 | Pachinko & Karaoke Amusement (パチンコ遊技機・カラオケ・大衆アミューズメント) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 57 | Paints, Inks & Functional Adhesives (塗料・インキ・接着剤・コーティング材料) | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 58 | Paper, Packaging, Containers & Printing (紙パルプ・包装容器・印刷) | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 59 | Passenger Cars, Commercial Vehicles & Motorcycles (乗用車・完成車・商用トラック・二輪) | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | Plant, Electrical & Telecom Engineering (設備工事・エンジニアリング・インフラ工事) | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 61 | Regional & Provincial Banking Groups (地域ブロック中核地銀) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| 62 | Restaurants & Food Service (外食・フードサービス) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| 63 | Seasonings, Packaged Foods & Processed Meats (調味料・即席麺・食肉・調理加工食品) | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | Securities, Exchanges & Investment (証券・取引所・投資育成) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 65 | Security, Facility Management & SMB Recurring Services (警備・施設保守・中小企業ストック支援) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 66 | Semiconductor & Advanced Electronics Materials (半導体プロセス材料・先端電子ケミカル) | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 67 | Semiconductors & Semiconductor Manufacturing Equipment (半導体デバイス・半導体製造装置・SPE) | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 68 | Specialized Trading - Pharma, Medical & Daily Goods (専門商社-医薬品・医療機器・日用品卸) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| 69 | Specialized Trading - Tech, Electronics & Materials (専門商社-エレクトロニクス・機械・化学素材) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| 70 | Specialty Biopharma & Biotech Platforms (スペシャリティ創薬・眼科・バイオテック) | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 71 | Stationery, Office & Household Brands (文具・オフィス家具・楽器・生活用品) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 72 | Supermarkets, GMS & Discount Stores (スーパー・総合小売・ディスカウントストア) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 0 |
| 73 | Surfactants, Oleochemicals & Fine Chemicals (界面活性剤・油脂化学・高機能ファインケミカル) | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 74 | Tech Investment, Strategic Holding & Capital Allocators (テック投資・戦略持株会社・資本アロケーター) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 75 | Telecom & Mobile Operators (通信キャリア・通信インフラ・情報流通) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 76 | Theme Parks, Resorts & Hotels (テーマパーク・リゾートホテル・複合レジャー) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 77 | Tires & Rubber Products (タイヤ・ゴム製品) | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 78 | Warehousing, Harbor & Logistics Terminal (倉庫・港湾運送・ターミナル物流) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |


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


## Phase 0 mapping gates
- 17 sectors with ≥1 covered subsector: True (need True)
- Subsectors with zero aggregation weight: 0
