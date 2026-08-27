# task3 — F2-a: コーパス頻度表ビルダー（train 域記事のみ・リーク対策込み）

対象 issue: #19（A 項目・F2 の前半）
実行機: ローカル（`/home/sarashin/work/beme-19`）
依存: task1（被覆率の実測が PLAN §3.5 分岐 C の判定に要る）
種別: 実装 + データ生成

---

## 0. 目的

F2 の 3 特徴が読む外部知識、すなわち **train 域の Wikipedia 記事だけから数えた (読み, 表記) 頻度表**を作る。

このタスクの成否は精度ではなく**リークしていないこと**で決まる。d1 の split は記事単位のハッシュで決まっており、同じコーパスを流すこのビルダーが dev/test 記事を数えてしまえば、以降の全実験の数値が無効になる。

---

## 1. 必ず読むもの

- `gh issue view 19`（コメント §4 のリーク対策の指示。**これが正本**）
- `docs/missions/issue19/PLAN.md` §2.1（split 判定の再現性を実測済みであること）・§3.4（3 重のリーク対策）・§3.5 分岐 C（被覆率が低かったときの対応）
- `docs/missions/issue19/s-measurement-report.md`（task1 の成果。被覆率の実測値）
- `scripts/build_d1_data.py` — **最重要**。特に:
  - `article_split()`（56-62 行）— この関数を **import して使う。写経しない**
  - `iter_hf_articles()`（112-121 行）— dataset / config / revision の流し方
  - `node_reading()`（80-86 行）— MeCab ノードから読みを取る（カタカナ→ひらがな・かな以外は None）
  - `load_exclusion_terms()` / `overlaps_evaluation()`（98 行付近・412-414 行）— 評価スイート重複記事の除外
  - `build()`（379-393 行）— `GenericTagger(f"-r /dev/null -d {ipadic.DICDIR}")` の作り方
- `data/d1/manifest.json` の `source` ブロック（dataset / config / **revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`** / 走査バイト数）
- `src/beme/matrix_cost.py` — 表のバイナリ形式・`load()` の書き方の**先例**。F2 の表もこの作法に合わせる
- `scripts/count_corpus_tokens.py`・`scripts/audit_leakage.py`（manifest / SHA-256 の作法）

---

## 2. 作るもの

### 2.1 `src/beme/corpus_freq.py`（表の読み書き）

`src/beme/matrix_cost.py` に倣う。

```python
DEFAULT_CORPUS_FREQ_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus_freq_v1.bin"
MAGIC = b"BEMECF1"   # 形式は自由。マジック + version + カウント数のヘッダを持つこと
```

`CorpusFrequencyTable` が持つべき API（task4 の特徴実装がこれだけを使う）:

| メソッド | 返すもの |
|---|---|
| `pair_count(reading, surface) -> int` | `c(r, s)` |
| `reading_count(reading) -> int` | `c(r) = Σ_s c(r, s)` |
| `surface_count(surface) -> int` | `c(s)` |
| `total_tokens` (property) | `N` |
| `classmethod load(path=DEFAULT_CORPUS_FREQ_PATH)` | 表。**ファイルが無ければ明示的な例外**（黙って空表を返さない） |
| `classmethod from_counts(pairs, surfaces, total)` | テスト用のインメモリ構築 |
| `dump(path)` | 決定的なバイト列を書く（同じ入力から同じ SHA-256 が出ること） |

形式は自作バイナリでも `msgpack` 風でもよいが、**依存を増やさない**こと（標準ライブラリ + 既存依存のみ）。34 万ペア程度なので、UTF-8 文字列 + `struct` のフラットな配列で十分。ロード時間は 1 秒未満を目標。

### 2.2 `scripts/build_corpus_freq.py`（ビルダー）

CLI:

```
--output PATH            既定 data/corpus_freq_v1.bin
--manifest PATH          既定 data/corpus_freq_v1.manifest.json
--dataset / --config / --revision   既定は data/d1/manifest.json の source と同じ値
--target-bytes INT       走査する UTF-8 バイト数（既定 104857600 = d1 と同じ 100MiB）
--max-articles INT       スモーク用
--split-seed INT         既定 20260822
--articles-jsonl PATH    テスト用のローカル記事ソース（build_d1_data と同じ形式）
--d1-dir PATH            既定 data/d1。リーク監査に使う
```

処理:

1. `from build_d1_data import article_split, node_reading, iter_hf_articles, iter_jsonl_articles, load_exclusion_terms, overlaps_evaluation` — **`scripts/` を sys.path に足して import する。ハッシュ式を写経しない**（先例: `scripts/evaluate_d1.py:31` の `from train_d1_qlora import prompt_text`）
2. 記事を 1 本ずつ流す。次の 2 条件のどちらかに当たった記事は**スキップして数を記録する**
   - `article_split(article.article_id, split_seed) != "train"` → `dev_test_articles_skipped += 1`
   - `overlaps_evaluation(article.text, exclusion_terms)` → `evaluation_overlap_articles_skipped += 1`
   （**両方必ず適用する**。2 つ目を落とすと d1 は汚さずに AJIMEE / handmade スイートを汚す。PLAN §3.4)
3. 残った記事を `GenericTagger` で解析し、各ノードについて
   - `reading = node_reading(node)`。`None`（読みが取れない・かな以外）ならスキップ
   - `pair_counts[(reading, node.surface)] += 1`・`surface_counts[node.surface] += 1`・`total_tokens += 1`
4. `--target-bytes` に達したら停止（`fetch_corpus.py` と同じ数え方: **スキップした記事のバイト数も走査量に数える**。そうしないと走査窓が d1 と揃わない。どちらにしたかを manifest に記録すること）
5. 表を `--output` に dump し、manifest を書く

### 2.3 manifest（`data/corpus_freq_v1.manifest.json`・**これはコミットする**）

```json
{
  "schema_version": 1,
  "created_on": "2026-08-..",
  "source": {"dataset": "...", "config": "20231101.ja", "revision": "b04c8d1...", "utf8_bytes_scanned": 0},
  "split": {"algorithm": "sha256(seed:article_id) modulo 10000", "seed": 20260822, "train_upper": 7700},
  "articles": {
    "seen": 0,
    "train_counted": 0,
    "dev_test_skipped": 0,
    "evaluation_overlap_skipped": 0
  },
  "counts": {"total_tokens": 0, "unique_pairs": 0, "unique_surfaces": 0, "unique_readings": 0},
  "leak_audit": {
    "d1_dev_test_article_ids": 0,
    "counted_article_ids_intersecting_d1_dev_test": 0,
    "passed": true
  },
  "coverage": {
    "d1_test_gold_pairs": 0,
    "d1_test_gold_pairs_seen": 0,
    "d1_test_gold_pair_coverage": 0.0,
    "d1_test_gold_surfaces_seen": 0,
    "d1_test_gold_surface_coverage": 0.0,
    "d1_train_gold_pair_coverage": 0.0
  },
  "table_sha256": "..."
}
```

- `leak_audit.counted_article_ids_intersecting_d1_dev_test` は **0 でなければならない**。0 でなければビルダーは非ゼロ終了する
- `coverage` は task1 の `coverage_pairs.jsonl` を読んで埋める（d1 を再走査しない）。これが **PLAN §3.5 分岐 C の判定値**である

### 2.4 `tests/test_corpus_freq.py`

- `CorpusFrequencyTable` の round-trip（`from_counts` → `dump` → `load` で完全一致・SHA-256 が決定的）
- ファイルが無いパスで `load()` すると明示的な例外が出ること（空表にならないこと）
- **リーク検査**: `--articles-jsonl` に手作りの記事を 3 本（train バケットに落ちるもの・dev に落ちるもの・test に落ちるもの）与え、train バケットの記事のトークンだけが数えられていること。`article_split` の実際の値でバケットが決まるので、テストは **article_id を総当たりで探して 3 バケット分を作る**こと（ハッシュ値をハードコードしない）
- **評価スイート除外の検査**: `data/evaluation_cases.jsonl` 由来の文字列を本文に含む記事が除外されること
- `data/d1/*.jsonl` にも `data/corpus_freq_v1.bin` にも依存しないこと（どちらも gitignore で CI に存在しない）

---

## 3. 実行

```bash
# スモーク（記事 200 本）
uv run python scripts/build_corpus_freq.py --max-articles 200 \
  --output /tmp/claude-1000/cf-smoke.bin --manifest /tmp/claude-1000/cf-smoke.json

# 本番（d1 と同じ 100MiB 窓）
uv run python scripts/build_corpus_freq.py \
  --output data/corpus_freq_v1.bin --manifest data/corpus_freq_v1.manifest.json
```

**PLAN §3.5 分岐 C の判定**: manifest の `coverage.d1_test_gold_pair_coverage` を見る。

| 実測値 | 対応 |
|---|---|
| ≥ 0.70 | そのまま task4 へ |
| 0.50〜0.70 | `--target-bytes 419430400`（4 倍）で再ビルドして再測。改善しなければ task4 に「4 特徴目 `lex.corpus_surface_unseen` を足す」と申し送る |
| < 0.50 | 4 倍再ビルド **かつ** 4 特徴目の申し送りを両方行う。それでも < 0.50 なら task7 に「F2 の効く範囲は d1 の半分未満」と記録する申し送りを書く |

いずれの場合も **task4 は実行する**。「効かなかった」も測定結果であり、F0 の教訓（新情報ゼロだったから効かなかった）と区別できるのは被覆率の実測があるときだけである。

`.gitignore` に `data/corpus_freq_v1.bin` を足すこと（manifest はコミットする）。

---

## 4. 完了条件

- [ ] `src/beme/corpus_freq.py` が §2.1 の API を持ち、決定的に dump/load できる
- [ ] `scripts/build_corpus_freq.py` が §2.2 の CLI で動く
- [ ] `tests/test_corpus_freq.py` が §2.4 の 5 項目を検査している（**リーク検査を含む**）
- [ ] `uv run pytest -q` が **638 + 新規テスト数 passed / 2 failed**（失敗は `tests/test_audit_leakage.py` の 2 件のみ）
- [ ] `data/corpus_freq_v1.bin` が生成され、`data/corpus_freq_v1.manifest.json` の `leak_audit.passed` が `true`
- [ ] manifest の `coverage.d1_test_gold_pair_coverage` が埋まっており、PLAN §3.5 分岐 C の判定が報告に書かれている
- [ ] `.gitignore` に表本体が追加されている
- [ ] `git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py` が **空**
- [ ] git commit / push をしていない

---

## 5. 検証コマンド

```bash
cd /home/sarashin/work/beme-19

uv run pytest -q
uv run pytest -q tests/test_corpus_freq.py -v

# リーク監査の独立確認（ビルダーとは別経路で数える）
uv run python -c "
import json, sys, hashlib
sys.path.insert(0,'scripts')
from build_d1_data import article_split
ids=set()
for name in ('dev','test'):
    for line in open(f'data/d1/{name}.jsonl'):
        ids.add(json.loads(line)['article_id'])
bad=[a for a in ids if article_split(a,20260822)=='train']
print('d1 dev/test articles:',len(ids),'/ misclassified as train:',len(bad))
assert not bad
m=json.load(open('data/corpus_freq_v1.manifest.json'))
print('leak_audit:',m['leak_audit'])
print('coverage :',m['coverage'])
assert m['leak_audit']['passed'] and m['leak_audit']['counted_article_ids_intersecting_d1_dev_test']==0
"

# 表の SHA-256 と manifest の一致
sha256sum data/corpus_freq_v1.bin
uv run python -c "import json;print(json.load(open('data/corpus_freq_v1.manifest.json'))['table_sha256'])"

git diff --check
git status --short
```

---

## 6. 制約（厳守）

- **`src/beme/lattice.py` / `src/beme/converter.py` 変更禁止**（探索コア）
- このタスクでは `src/beme/features.py` も変更しない（特徴の実装は task4）
- **`article_split` のハッシュ式を写経しない。** `build_d1_data` から import する。写経が唯一のリーク経路である
- **評価スイート除外（`overlaps_evaluation`）を落とさない。** 落とすと d1 は汚さずに AJIMEE / handmade を汚す
- 依存パッケージを増やさない（標準ライブラリ + 既存 `dependency-groups.dev` のみ）
- `git commit` / `git push` をしない。作業はこの worktree の中だけ。`git stash` を使わない
- `tests/test_audit_leakage.py` の既存 2 件の失敗は**直さない**（環境固有。PLAN §2.7）
- 表本体は gitignore。manifest はコミット対象（ただし commit は Hermes が行う）
- 分岐 C の閾値（0.70 / 0.50）を事後に動かさない
- 日本語で報告。数値と結論を先頭に。特に `leak_audit` と `coverage` の実測値を必ず含める
