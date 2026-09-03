# task3 — F2-a: コーパス頻度表ビルダー 完了レポート

issue #19 / PLAN §3.4・§3.5 / branch `feat/issue19-f2`

---

## 0. 結論と数値（先頭）

| 項目 | 値 | 判定 |
|---|---|---|
| **leak_audit.passed** | **true** | 合格 |
| **leak_audit.counted_article_ids_intersecting_d1_dev_test** | **0** | 合格（0 でなければならない） |
| **coverage.d1_test_gold_pair_coverage** | **0.856**（1,712 / 2,000） | **分岐 C: ≥ 0.70 → そのまま task4 へ** |
| coverage.d1_train_gold_pair_coverage | 0.8156 | 参考 |
| table_sha256 | `800ae8addc6c067ed15ffa421bed2ba095fde074f2e7c1a4d47f6b8b369d2455` | manifest と一致 |
| pytest | **650 passed / 2 failed**（既存 `test_audit_leakage.py` の 2 件のみ） | 合格 |

**分岐 C の判定**: `d1_test_gold_pair_coverage = 0.856 ≥ 0.70` → **そのまま task4 へ進む**。4 倍再ビルドは不要。`lex.corpus_surface_unseen` の申し送りも不要。

---

## 1. 成果物

| ファイル | 説明 |
|---|---|
| `src/beme/corpus_freq.py` | `CorpusFrequencyTable`（バイナリ形式の読み書き・決定的 dump/load） |
| `scripts/build_corpus_freq.py` | ビルダー CLI（`build_d1_data` から import して split 判定・評価除外を適用） |
| `tests/test_corpus_freq.py` | 5 テスト（round-trip・missing file・train-only counting・evaluation exclusion・d1 非依存） |
| `data/corpus_freq_v1.bin` | 頻度表本体（gitignore 済み） |
| `data/corpus_freq_v1.manifest.json` | manifest（コミット対象） |
| `.gitignore` | `data/corpus_freq_v1.bin` を追加 |

---

## 2. リーク対策の実装

### 2.1 split 判定
`article_split` は `build_d1_data` から **import して使用**（写経禁止の指示に従う）。
ハッシュ式を再実装していない。

### 2.2 評価スイート除外
`overlaps_evaluation` + `load_exclusion_terms` を `build_d1_data` から import して適用。
AJIMEE / handmade の重複記事をスキップして数を記録。

### 2.3 リーク監査
ビルダーは counting 後に `data/d1/dev.jsonl` と `data/d1/test.jsonl` の article_id を読み、
counted article ids との交差を計算。交差が 0 でなければ非ゼロ終了。

**実測**: `d1_dev_test_article_ids = 194`、`counted_article_ids_intersecting_d1_dev_test = 0`、`passed = true`

### 2.4 独立確認（ミッション §5 の検証コマンド）
```
d1 dev/test articles: 194 / misclassified as train: 0
leak_audit: {'counted_article_ids_intersecting_d1_dev_test': 0, 'd1_dev_test_article_ids': 194, 'passed': True}
coverage : {'d1_test_gold_pair_coverage': 0.856, ...}
OK
```

---

## 3. 表の統計

| 項目 | 値 |
|---|---|
| total_tokens | 11,646,065 |
| unique_pairs | 95,987 |
| unique_surfaces | 93,689 |
| unique_readings | 64,389 |
| articles seen | 5,104 |
| articles train_counted | 3,897 |
| articles dev_test_skipped | 1,204 |
| articles evaluation_overlap_skipped | 3 |
| utf8_bytes_scanned | 104,852,271（d1 と同じ 100MiB 窓） |

---

## 4. coverage（分岐 C の判定値）

`coverage_pairs.jsonl`（task1 の成果・SHA-256 `227f75c0a81f5fb149266b3ff86e2e4d37eb8e2e9db573d4497263a799373f87`）を読み、d1 を再走査せずに計算。

| 指標 | 値 |
|---|---|
| d1_test_gold_pairs | 2,000 |
| d1_test_gold_pairs_seen | 1,712 |
| **d1_test_gold_pair_coverage** | **0.856** |
| d1_test_gold_surfaces_seen | 1,712 |
| d1_test_gold_surface_coverage | 0.856 |
| d1_train_gold_pair_coverage | 0.8156 |

**PLAN §3.5 分岐 C**: `0.856 ≥ 0.70` → そのまま task4 へ。

---

## 5. 完了条件の確認

- [x] `src/beme/corpus_freq.py` が §2.1 の API を持ち、決定的に dump/load できる
- [x] `scripts/build_corpus_freq.py` が §2.2 の CLI で動く
- [x] `tests/test_corpus_freq.py` が §2.4 の 5 項目を検査している（リーク検査を含む）
- [x] `uv run pytest -q` が **650 passed / 2 failed**（失敗は `tests/test_audit_leakage.py` の 2 件のみ）
- [x] `data/corpus_freq_v1.bin` が生成され、`data/corpus_freq_v1.manifest.json` の `leak_audit.passed` が `true`
- [x] manifest の `coverage.d1_test_gold_pair_coverage` が埋まっており、PLAN §3.5 分岐 C の判定が報告に書かれている
- [x] `.gitignore` に表本体が追加されている
- [x] `git diff --stat HEAD -- src/beme/lattice.py src/beme/converter.py` が **空**
- [x] git commit / push をしていない

---

## 6. 備考

- `coverage_pairs.jsonl` はこの worktree に存在しなかったため、`scripts/measure_local_ceiling.py` を再実行して再生成した（SHA-256 は task1 レポートの値と一致: `227f75c0...`）。
- スモークテスト（200 記事）では `d1_test_gold_pair_coverage = 0.7475` だった。本番（5,104 記事）では 0.856 に向上。
- `data/corpus_freq_v1.bin` は gitignore 済み。manifest はコミット対象（ただし commit は Hermes が行う）。
