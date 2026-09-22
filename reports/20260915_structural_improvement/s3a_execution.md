# S2b完了確認・S3a実行記録

- 実施日: 2026-09-16
- 対象: 修正済み作業ツリー（未commit差分を保持）
- S2b判定: **範囲内完了（PASS）**
- S3a判定: **型付き分布結果・理由・試行記録の統一（PASS）**

## S2b完了確認

既存の[S2b実行記録](s2b_execution.md)に記録された実装・全体検証結果を確認し、今回の確認では
入力契約、PIT、read-only所有権、V2分布sourceの境界を再実行した。

| 検証 | 結果 |
|---|---:|
| S2b input contract / S2a boundary / PIT | **25 passed** |
| V2 / Stage A-C / auditor / gap distribution 回帰 | **110 passed** |
| 既存S2b実行記録の全体検証 | **673 passed / 18 warnings** |

S2bの入力版付け・PIT入力分離・read-only所有権・主要4経路の切替は完了している。旧互換entry、
`data/cache.py` shim、`models/blpx.py`再export、公開dict wrapperの撤去と、macro/ADRの
`available_at`厳密化はS2全体の残件として維持する。これらを残したままS2全体完了とは扱わない。

## S3aで実施した変更

| 境界 | 変更 |
|---|---|
| domain | `domain/distribution.py`を正本にし、`DistributionStatus`、`DistributionReason`、`DistributionAttempt`、`DistributionResolutionError`、`DistributionResult`を追加 |
| source | cache・on-demandの未設定、欠損、計算失敗、来歴拒否を理由コードへ変換。raw alertはログ・互換出力として保持 |
| fallback | `FallbackPolicy`が全sourceの試行順と結果を記録し、`REJECTED` / `PROVENANCE_REJECTED`から監査失敗を決定。alert文字列の検索を廃止 |
| MH / decision | `DistributionResolutionError`を理由コード付きで伝播し、multi-horizonとsingle-horizonのflat化・diagnosticsを同じ契約で組み立て |
| compatibility | `is_available`、`is_flat`、`alerts`、`flat_decision`、既存source importは一時互換として保持。内部判定はtyped status/reasonを使用 |

`[FATAL]`や`"provenance"`という文字列を使った理由推定はfallback/decision経路から撤去した。
rawデータ由来のalert分類はI/O境界に限定し、監査の数値・リーク検査自体は変更していない。

## 検証

| 検証 | 結果 |
|---|---:|
| S3a新規 + 既存source/S2b/PIT | **42 passed** |
| V2 integration/regression/Stage A-C/auditor | **110 passed** |
| `git diff --check` | PASS |
| `compileall` | PASS |
| Ruff（変更対象） | PASS |
| plan / ADR / architecture 文書検証 | **PASS（63 links）** |
| 全`tests/`（xdist 10 workers） | **677 passed / 17 warnings / 710.11秒** |

## 残件と次段階

S3aの対象は結果契約と理由伝播までであり、S3全体の完了ではない。次段階では、h=1・MH・公開
`compute_distribution`のresolver重複を一つの正規経路へ移し、利用者を切り替えたうえで不要な
compatibility wrapperを撤去する。gap生成の責務分割、bundle manifest、VaR部品整理はS3b/S3cで扱う。
