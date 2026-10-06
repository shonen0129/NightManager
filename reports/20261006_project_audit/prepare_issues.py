import json
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path('/Users/shonen/leadlag')
OUT = ROOT / 'reports/20261006_project_audit'
SHA = 'b5b901e6d467fb000263f9178722e3954d652720'
REPO = 'shonen0129/NightManager'
BASE = f'https://github.com/{REPO}/blob/{SHA}/'
report = (OUT / 'report.md').read_text()
findings = {f['id']: f for f in json.loads((OUT / 'findings.json').read_text())}
sections = dict(re.findall(r'^### (F\d+) (.*?)(?=^### F\d+|^## \d+|\Z)', report, re.M | re.S))

def links(match):
    label, url = match.groups()
    if url.startswith(('http://', 'https://')):
        return match.group(0)
    path = (OUT / url.split('#')[0]).resolve()
    if path.is_relative_to(ROOT) and path.exists() and not path.is_relative_to(OUT):
        return f'[{label}]({BASE}{quote(str(path.relative_to(ROOT)))})'
    return f'{label}（ローカル監査証跡 `{url}`）'

def evidence(ids):
    chunks = []
    for fid in ids:
        chunks.append('### ' + fid + ' ' + re.sub(r'\[([^\]]+)\]\(([^)]+)\)', links, sections[fid].strip()))
    return '\n\n'.join(chunks)

COMMON = '''## 検証・変更の制約

- 2026-10-06監査では全984テストと本番lint・型・import契約が成功した。以下は成功した既存検証の外側に残る課題。
- 実行コード・挙動に関わる設定を修正した場合、対象回帰→`tests/`全体→必須compileall・CI対象チェックを実行し、プロセス全体の停止期限を設ける。文書だけなら参照・構造・意味の整合を検証する。
- 既存ユーザー変更、strictly historical計算、2010–2014固定prior、監査・risk fail-closed、当日cache→on-demand→flat、durable reconciliationを保持する。互換wrapperや正本の二重化で解決しない。
- 本issueの登録は修正・本番反映・発注の実行承認ではない。研究上の採否とproduction変更を分ける。
'''

# One issue per coherent implementation/acceptance boundary, not one per audit row.
specs = [
 ('pnl', 'P1', ['F01','F18','F19'], '持越し損益・終端在庫・執行turnoverを在庫会計の正規契約へ揃える', [
  'close→翌09:10の持越し損益を寄付gapと寄付→09:10に分けて手計算と一致させる。翌日flat、long/short、逆方向gap、週末・連休を含める。',
  'initial holdingsとterminal policyを明示し、終端残在庫・評価時刻・cash・費用を出力する。artifact変更で在庫を暗黙resetしない連続replayを作る。',
  '主たる執行turnover/volumeをopening/closingのinventory flowから計算し、target-weight turnoverとmodel/effective exposure・1/2係数を区別する。',
  '正規入口→価格区間抽出→PnL→成果物までの回帰で監査の合成再現を固定する。',
  '修正後、baseline・ML paired比較・carry評価・VaR/ESを同じ損益定義で再計算し、旧結果との差と適用範囲を保存する。'],
  '旧#12のflat移行決済cost修正は維持する。今回は寄付→09:10、配列終端、報告turnoverの追加欠陥を扱う。再評価は#23/#25/#27へ接続する。'),
 ('secrets', 'P1', ['F02'], 'backtest設定保存とbroker HTTP例外から認証情報を除去する', [
  'artifact保存用のallowlistでAppConfigからpassword/token/第二passwordを除外し、再利用可能な非秘密設定とconfig hashを保存する。',
  'API境界でHTTP例外を安全な例外へ変換し、URL query・認証payload・raw response等の秘密をlog/summaryへ出さない。',
  '合成secretをCLI→SQLite、broker login/order/healthの失敗→logへ通し、全出力に残らない回帰を追加する。',
  '既存artifact/logの影響範囲と取扱いを記録し、過去の値が実資格だった場合の失効・再発行状態を確認する。実credentialをissueへ転載しない。'],
  'capture側のsafe error/allowlist診断（#29/#30のPR）は維持する。API全体とbacktest storeは別の未保護境界。'),
 ('adr', 'P1', ['F05','F22'], 'ADR日次producerを運用層へ分離し更新復旧とML適用状態を可視化する', [
  '日次更新の失敗原因をログ・入力・依存から特定し、当日のADR日付/coverageを満たすproducerを復旧する。旧値の当日流用で通さない。',
  '安定したADR生成を本番data/operational層へ置き、scheduled updaterからresearch実験スクリプトへの直接依存を除去する。',
  'scheduled diagnosticsを含む運用入口を依存検査の対象にし、本番wheelにresearchを混入させず運用依存を満たす。',
  'ADR複数artifactを単一version/manifestでatomic publishし、途中失敗や同時読込で新旧混在しない。',
  'MLのenabled/applied/skipped/rejected、reason、feature freshness/coverageを成功・skip双方で構造化して保存・集計する。on-demand、flat、PIT multiplierとは区別する。',
  '更新成功/失敗、stale/欠損ADR、ML適用/非適用をpreflight→decision→summaryまで回帰検証する。'],
  '#27の市場→shadow受入、#23のpaired forward観測に適用状態を渡す。全体deadlineは別issueで扱う。旧#19のwheel研究除外は完了済みとして保持する。'),
 ('deadlines', 'P2', ['F06'], '全scheduled producerと長時間補助CLIへ全体deadline・single-flightを適用する', [
  'update_market_data、distribution diagnostics、full backtest、unit-only runner等の長時間入口を列挙し、全体deadlineと終了猶予を設定する。',
  'cacheを更新する非発注producerにもsingle-flight/leaseと工程budgetを適用し、timeoutを正常終了にしない。',
  'stallした子/孫processを期限内に回収し、leaseを解放してexit code・phase status・失敗artifactを整合させる。',
  'scheduled entryがguard/deadlineを使う契約検査と、次回実行で正常復旧する回帰を追加する。'],
  '旧#17で整備されたdecision/gap/close/captureの安全策は保持する。threadの待機timeoutをprocess全体期限として扱わない。'),
 ('profitability', 'P1', ['F07','F09','F10'], '09:10戦略のtarget・実費・執行制御を揃えて収益とリスクを再評価する', [
  '09:10測定/proxyとopen→close fallbackを成績・coverage・target schemaで分離し、全評価営業日を保持する。不足をdropして成功評価にしない。',
  'frozen quote/provider available_at/5分足の時刻意味・履歴修正を固定し、実行可能価格とHigh/Low midpointの限界を明示する。',
  'order/fill ID、quote、数量/lot、cash/在庫、未約定/部分約定、全fee/financing/borrow/reverseを照合し、実費とcounterfactual費用を別表示する。',
  '未停止model reference系列と、risk STOP・actual-account evidence・reduction-only・在庫連続性を含む実行replayを別成果物として報告する。risk推定用系列を再帰的STOP系列へ置換しない。',
  'AUM・volume/depth・貸株可能量に対する容量、実効gross/net・実約定後β・集中を検証する。ドル中立だけでβ中立を認定しない。',
  '正規PnL修正後、rolling VaR/ES、最悪日寄与、費用stress、尾部標本の不確実性を再評価し、既存stop閾値を維持する。',
  '新規パラメータ/制約を提案する場合は別の事前実験として感度・DSR・walk-forward OOSを満たす。既知期間の再評価を新しいOOSと呼ばない。'],
  '#23は現行MLの固定artifact・250日forward gateの正本として維持する。本issueは基礎戦略/執行評価の証拠契約を担う。実口座正本は#25、市場取得は#27。旧#24の1.30へのリスク低減は完了済みで、今回の2.1bp余裕を現在の閾値超過とは扱わない。'),
 ('period', 'P2', ['F11','F12'], 'V2評価入口でprior期間との分離と要求期間の交差を検証する', [
  '正規API/CLIで評価開始日2015-01-05以降を強制し、2010–2014の固定priorと評価期間の非交差を保証する。',
  '全データより前/後、空交差、start>end、NaT、空frameを明示拒否または規定の空結果とし、要求外の日を選択しない。',
  '要求期間・実際の期間・baseline期間をmanifestへ保存し、非営業日境界の既存挙動を保持する。',
  'resolver単体に加え正規run_v2_backtest/CLIで境界を回帰検証する。先頭1260行prior fallbackで解決しない。'],
  '固定2010–2014のfinite baseline検査は保持する。default開始日だけを期間分離の保証にしない。'),
 ('metrics', 'P2', ['F13','F14','F17'], '月次年率化・summary DD・欠損評価日のmetrics契約を統一する', [
  'frequencyとannualizationを正規契約へ揃え、monthly指定単独で245年率にならない。省略形を廃止するなら呼び出し元も更新する。',
  'summary writerのDD再実装を削除し、初期wealth1.0を含む共有計算からCSV/JSON/CLIで同じ値を出す。',
  'label unavailableをPnL前に判別し、zero exposureで損益0が確定する場合とactive exposureで不明な場合を分ける。',
  '全営業日coverage・invalid status・有効観測数を出力し、NaN日を暗黙dropして成功評価にしない。',
  '月次4観測、初日-10%/翌日0、flat+欠損、active+欠損を正規入口→summary/registryまで回帰検証する。'],
  '旧#13で修正済みの共有daily MDD/DSR年率変換を保持する。今回はfrequency省略形・writer再実装・欠損日の別経路を追跡する。'),
 ('minvar', 'P2', ['F15'], 'MinVarのbasket選択でlong_countとshort_countを正しく扱う', [
  'basket選択を一度だけ行い明示indicesでMinVarへ渡す、または非対称設定が非対応ならschemaで明示拒否する。',
  'long5/short3、対称5/5、銘柄数上限、side重複・空basketを正規モデル入口で検証する。',
  'モデルnet±0.05/gross≤2と既存監査の厳しい閾値を維持し、選択数とdecision summaryを一致させる。'],
  '現行productionの5/5には当該再現差がない。設定契約として許された非対称値の無視を修正する。'),
 ('registry', 'P2', ['F16','F28'], 'DSR入力とcomputed metricsを検証しstudy単位の探索履歴を追跡する', [
  'metric_status、全評価return系列の有限性、観測数T、frequency/年率係数、trials・試行間varianceを一体で検証し、不整合/invalidを拒否する。',
  'extra_metricsからcomputed fields/status/観測数を無条件上書きできなくし、必要なoverrideは明示した契約へ限定する。',
  '4returns/T1000、NaN、unknown frequency、invalid status、computed field上書きの回帰を追加する。',
  'study IDを仮説family単位で事前登録し、全候補/棄却/中断・選択時点・code/config/data/target/cost schema・IS/OOS/purgeを記録する。',
  '過去reportを根拠に探索索引を作り、不明試行はunknown/lower boundと明示する。架空試行を作らず既存legacy IDと追記訂正履歴を保持する。',
  '修正の影響を受ける既存DSRを識別して再計算またはinvalid/未確認とし、全DSRが誤りだと一括判定しない。'],
  '旧#13のannualized Sharpe/DSR変換と、helperの明示trials/study ID保持は既に整備済み。今回の入力整合と探索family集約は別の残件。'),
 ('providers', 'P2', ['F20'], '未接続providerの時刻・OHLC欠損契約を修正または撤去する', [
  '本番providerとして維持する利用意図を確認し、不要なら呼び出し元/testを整理して本番packageから撤去または研究へ移す。',
  '維持する場合、at以前の観測だけを返し観測時刻/source/freshnessを明示する。opening priceを09:10 priceとして返さない。',
  'daily OHLCは必須価格列だけで妥当性判定し、任意Volume欠損で正常行を削除しない。',
  '09:10=100/15:00=200、at以前の観測なし、正常OHLC/Volumeなしを回帰検証する。'],
  '現行本番fetcher/frozen quoteから未接続であり、現行liveへ未来価格が流れたというissueではない。不要層整理issueと対象が重なるため削除を二重実装しない。'),
 ('runtime_root', 'P2', ['F21'], 'wheelのpackage位置とdeployment runtime rootを分離する', [
  'package code位置からdata rootを推測せず、正規設定からdeployment rootを明示して解決する。',
  '相対config/model/ADR/macro cache/var出力を同一run rootへ解決し、旧root探索fallbackを追加しない。',
  'installed wheelをtmp deployment rootでread-only/offline実行し、default pathとartifact解決・出力先を検証する。helpだけのsmokeにしない。',
  'checkoutとinstalled packageで同じ設定/入力/成果物契約を満たすことをCIで検証する。'],
  '旧#19のresearch除外・wheel buildは完了済み。今回の再現は配置path契約のsimulationであり、現在のcheckout scheduler障害の原因とは認定しない。'),
 ('math', 'P3', ['F23'], '本番と研究のBLPX純粋計算を共有正本へ統合する', [
  'AST一致14組を呼び出し元/意味/入出力契約で再確認し、共有すべき純粋数学だけを正本化する。',
  'Tikhonov/PCA prior/confidence/asymmetric solve/signal等の研究呼び出しを同時更新し、モデル構成・研究変種は研究側に残す。',
  'finite/nonfinite/PSD/window入力と固定fixtureでbefore/after数値一致を検証する。',
  '互換wrapper・巨大framework・二重の正本を残さず、研究成果物と監査不変条件を保持する。'],
  'body一致はclass全体の同一性を意味しない。モデル改良/パラメータ変更をリファクタリングへ混ぜない。'),
 ('orchestration', 'P3', ['F24'], '日次実行・close・前処理・VaRの責務境界を明示して関数を分割する', [
  'quote/artifact/risk/state/outputの調停境界を列挙し、実装順を対象ADR/roadmapと照合する。',
  '入力確定、純粋計算、判断plan、broker副作用、reconciliationをrun-owned contractで分割する。preflight結果を型付きで渡す。',
  '同一fixtureでdecision/manifest/終了理由と副作用順序が一致し、監査失敗・STOP・partial/rejected・復旧の挙動を保持する。',
  '行数やAST branch数だけで完成を判定せず、境界のcontractと局所検証可能性を受入基準とする。'],
  'audit/SQLite/durable recoveryを簡略化対象にしない。損益・ADR状態等の実害修正を先行し、巨大汎用frameworkへ置換しない。'),
 ('cleanup', 'P3', ['F25'], '未使用の本番層・互換wrapper・reports依存のtest watchdogを整理する', [
  '実呼び出しと研究用途を再確認し、未使用CostCalculator、研究用convex optimizer等を撤去/研究移管する。',
  'gap/PnL/研究ML wrapperの呼び出し元・設定・testを正規APIへ同時更新し、旧名再公開や移行wrapperを残さない。',
  '常用watchdogをscripts/toolsの正規入口へ配置し、通常test runnerがreports内スクリプトを実行しないようにする。',
  '研究成果物は保持し、注文状態/入力型/GapStore transaction等の安全上必要な分割は保持する。',
  '正規runnerで全testsを実行し、未使用symbol/禁止参照の再混入を検査する。'],
  'providerの時刻欠陥は専用issueで扱う。移動だけを目的としたディレクトリ増設や新しい互換層を作らない。'),
 ('harness', 'P2', ['F26','F27'], 'AGENTS・Skill・IDE手順・運用仕様を現行契約へ整合する', [
  '日次手順の全square/overnight禁止を解決済みcarry設定・close処理へ整合させ、正しい残在庫・照合/復旧を説明する。',
  'V2のleakage/numerical/fallback/実行gateを実施項目・保証外・停止結果と対応付け、技術仕様の撤去済みclass/config参照を更新する。',
  'leak-auditのflat挙動、experiment-designの明示trials/study ID、debugging参照、Devin/Windsurfの旧path/存在しないtestを現行契約へ更新する。',
  'AGENTSのrunner説明をfeatures/regressionを含む現行実装へ更新し、不採用結果はreports/graveyardへ置く。Skillへ実験履歴を蓄積しない。',
  'IDE workflowから正本Skillへの短い参照に寄せ、current文書のliteral path/公開symbolと保有・fallback等の意味を検証する。歴史reportへ現行APIを強制しない。',
  '文書/Skillのみの変更は構造・参照・意味の整合を検証し、validator等のコードを変える場合は全tests/CIも実行する。'],
  '旧#21はREADME/ARCHITECTURE/roadmapの整合として完了済み。今回は運用文書・Skill/IDEを追加対象とする。現runnerからregressionが漏れているとは指摘しない。'),
 ('proxy', 'P2', ['F30'], 'US pre-inception proxyを上場前に限定し上場後欠損を品質異常として扱う', [
  'XLC/XLRE/MTUM/VLUE/USMVのproxyを明示inception期間に限定し、上場後の取得欠損をproxyで埋めない。',
  '研究用pre-inception proxyのticker/期間/方法/sourceとtarget関係をprovenanceへ記録する。',
  '上場後欠損を品質statusへ伝播し、strict検証と当日cache→on-demand→flat/停止の既定規則で処理する。',
  '2026年XLC中間close欠損をstrict=Trueで正常値として受理しない回帰を追加し、US/JP休日整列とbaseline検査を保持する。'],
  '旧#7/#8の非対称休日/warm-up修正は維持する。研究baselineに必要な上場前proxyと実銘柄の現在の欠損を区別する。'),
 ('calendar', 'P2', ['F31'], 'static年表外でもJPX年末年始休場を営業日から除外する', [
  'JPX固有の1/2・1/3・12/31休場を恒常規則として年表から分離し、未知年の祝日データ不足を明示する。',
  '2024-12-31/2028-01-03を非営業日として返し、年境界のprevious/next sessionを検証する。',
  '2025–2027の既存祝日とUS/JP非対称対応・risk freshness照合を維持する。'],
  '現行2026のstatic tableに今回の欠落はない。過去・2028以降の境界欠陥を扱う。JPX一次情報: https://www.jpx.co.jp/faq/others_general.html'),
]

plan = []
for key, priority, ids, title, checks, relation in specs:
    refs = []
    for fid in ids:
        for ref in findings[fid]['source_references']:
            s = f"- [{ref['path']}:{ref['line']}]({BASE}{quote(ref['path'])}#L{ref['line']})"
            if s not in refs:
                refs.append(s)
    body = f'''## 目的・追跡

2026-10-06プロジェクト全体監査の {', '.join(ids)} を、同じ修正/受入境界として追跡する。優先度 **{priority}**。親tracker: #22。
監査基準HEAD: `{SHA}`。
監査成果物 `reports/20261006_project_audit/` は現時点でローカル未commitのため、必要な所見を以下へ転載する。ソースへのリンクは監査時点のcommit固定。

## 根拠・再現・影響

{evidence(ids)}

## 完了条件

''' + '\n'.join('- [ ] ' + c for c in checks) + f'''

## 関連・範囲

{relation}

## 監査時点のコード

''' + ('\n'.join(refs) or '上記の保存済みreportを参照。') + '\n\n' + COMMON
    plan.append(dict(key=key, findings=ids, priority=priority, title=f'[{priority}] {title}', body=body))

existing = []
for number, ids, extra in [
 (27,['F03'],'10/5のallowlist診断で未読開示書類flag=1を確認した。認証前提を正規手順で解消後、次の取引日read-only preflight→capture18/18→gap/shadow同一snapshot ID→受入reportを確認する。今回の追記はcontrolled live/書類操作/発注の承認ではない。'),
 (25,['F04'],'既存コメントどおり、まず実口座ledger正本・一対一execution identity・評価価格・外部cash flow・全費用・照合責任を確定する。proxy PnLの架空producerで解除しない。市場→shadowは独立して部分受入可能だがrisk込みはBLOCKを維持する。'),
 (23,['F08'],'250日gateの閾値を維持し、269日retrospectiveや混在artifactをforwardへ数えない。損益区間・基礎戦略評価の新規issueで再評価が必要になった場合も、既知期間の訂正を新しいOOSへ数えない。ML enabledとapplied/skipをpaired観測の適格性と別に記録する。'),
 (18,['F29'],'Hosted CIは監査HEADで全step成功。残件は既存方針のrequired checks/rulesetの実効強制であり、CI新設ではない。設定作業はこのissue化の実施範囲に含めない。'),
]:
    existing.append(dict(number=number, findings=ids, comment=f'## 2026-10-06全体監査の追記（{", ".join(ids)}）\n\n監査HEAD: `{SHA}`。重複issueを作らず本issueで追跡する。\n\n{evidence(ids)}\n\n### 受入上の補足\n\n{extra}\n\nローカル監査成果物: `reports/20261006_project_audit/report.md`（未commit）。親tracker: #22。'))

all_ids = [fid for p in plan + existing for fid in p['findings']]
assert len(all_ids) == len(set(all_ids)) == 31
assert set(all_ids) == set(findings)
payload = dict(repository=REPO, audit_head=SHA, new_issues=plan, existing_comments=existing)
(OUT / 'issue_plan.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(new=len(plan), existing=len(existing), findings=len(all_ids), titles=[p['title'] for p in plan]), ensure_ascii=False, indent=2))
