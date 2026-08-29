読みました。実装はせず、計画のみ出力します。読み込み中に**仕様と実データのズレを 3 件**確認したので、計画に反映してあります（§0 に先出し）。

---

# task3 実行計画 — F2-a コーパス頻度表ビルダー

対象: worktree `/home/sarashin/beme-19`・branch `feat/issue19-f2`（`b6a219c`・クリーン）
正本: `docs/missions/issue19/task3-mission.md` / PLAN §2.1・§3.4・§3.5

---

## 0. 着手前に確定した事実（この worktree で実測。計画はこれに従う）

指示書の記述と実データが食い違う点が 3 つある。いずれも**指示書を機械的に守ると誤った成果物になる**ので、先に扱いを決めておく。

| # | 指示書の記述 | 実測 | 本計画の扱い |
|---|---|---|---|
| 1 | worktree は `/home/sarashin/work/beme-19` | 実体は **`/home/sarashin/beme-19`**（`data/d1/*.jsonl` は `/home/sarashin/beme-d1/repo/` への symlink） | 全コマンドを `/home/sarashin/beme-19` 基準で書く |
| 2 | 完了条件「`638` + 新規 passed / 2 failed」 | `pytest --collect-only` = **647 collected**（= 645 passed / 2 failed）。638 は task1 着手**前**の値で、task1 が 7 件追加している | 完了条件を **`645 + 新規テスト数 passed / 2 failed`** に読み替える（§5） |
| 3 | 「`coverage` は task1 の `coverage_pairs.jsonl` を読んで埋める」 | 同ファイルは `{reading, surface, count}` のみで **split フィールドが無い**（train+test 混合・10,342 ユニーク / 出現 12,000）。manifest が要求する `d1_test_gold_pair_coverage`（＝分岐 C の判定値）と `d1_train_gold_pair_coverage` を**このファイル単体では分離できない** | §2.4 の方式（d1 JSONL の `reading`/`positive` から split 別に再構成 + coverage_pairs.jsonl との完全一致 assert）を採る |

#3 の裏取りは済んでいる。`data/d1/{train,test}.jsonl` の各行の `(reading, positive)` の多重集合は、`coverage_pairs.jsonl` と**ユニーク 10,342 件・全カウント完全一致**（`merged == ref` が True）。つまり task1 の「gold ペア」は**ケース単位の (読み, positive)** であり、d1 JSONL のフィールドを読むだけで split 別に再現できる。指示書の「d1 を再走査しない」の趣旨は「lattice/converter を張り直す高コストな再測定をするな」であり、JSONL のフィールド読み出しはこれに反しない。

さらに、**分岐 C の判定値には構造的な上限がある**ことを記録しておく（§4.1）。

---

## 1. 成果物一覧

| パス | 種別 | git |
|---|---|---|
| `src/beme/corpus_freq.py` | 表の読み書き（`CorpusFrequencyTable`） | commit 対象 |
| `scripts/build_corpus_freq.py` | ビルダー CLI | commit 対象 |
| `tests/test_corpus_freq.py` | テスト（表 round-trip + リーク検査） | commit 対象 |
| `data/corpus_freq_v1.bin` | 表本体（約 5〜8 MB 見込み） | **gitignore** |
| `data/corpus_freq_v1.manifest.json` | manifest（SHA-256・記事数・被覆率・リーク監査） | **commit 対象** |
| `.gitignore` | `data/corpus_freq_v1.bin` を 1 行追加 | commit 対象 |

`src/beme/features.py` は**触らない**（特徴実装は task4）。`src/beme/lattice.py` / `converter.py` も触らない。

---

## 2. 各ファイルの設計

### 2.1 `src/beme/corpus_freq.py`

`src/beme/matrix_cost.py` の作法に合わせる（モジュール定数 → `@dataclass(frozen=True, slots=True)` → `classmethod load()` → `__all__`）。標準ライブラリ `struct` / `pathlib` / `dataclasses` のみ。**依存追加なし**。

```python
MAGIC = b"BEMECF1"
VERSION = 1
DEFAULT_CORPUS_FREQ_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus_freq_v1.bin"
_HEADER = struct.Struct("<7sBHHQIII")   # magic, version, reserved, reserved2, total_tokens, n_readings, n_surfaces, n_pairs
```

**バイナリ形式**（すべてリトルエンディアン・固定長 + ブロブ）

| 区画 | 内容 |
|---|---|
| header | `MAGIC`(7B) + `VERSION`(u8) + `reserved`(u16=0) + `reserved2`(u16=0) + `total_tokens`(u64) + `n_readings`(u32) + `n_surfaces`(u32) + `n_pairs`(u32) |
| readings blob | `sorted(readings)` を UTF-8 で連結。直前に `u32` 長配列（`n_readings` 個）を置く |
| surfaces blob | 同上（`sorted(surfaces)`・`u32` 長配列 + UTF-8 連結） |
| reading counts | `u64 × n_readings`（`c(r) = Σ_s c(r,s)`。ビルド時に確定させ保存する。load 時に再計算しない） |
| surface counts | `u64 × n_surfaces` |
| pairs | `(reading_index u32, surface_index u32, count u32) × n_pairs`。`(reading_index, surface_index)` の昇順 |

**決定性**: 文字列は `sorted()`（コードポイント順）で並べ、index は必ずその順序から引く。dict のイテレーション順序に一切依存させない。同じ入力から同じバイト列 → 同じ SHA-256。ペイロード末尾に余剰バイトがあれば `ValueError`（`matrix_cost.load` の `stream.read(1)` チェックと同じ流儀）。

**サイズ/速度見積り**: ユニークペア 34 万想定で pairs 12B × 34万 ≒ 4 MB + 文字列ブロブ数 MB。load は `struct.unpack` 3 回 + `dict` 構築 34 万件で 0.3〜0.6 秒。目標 1 秒未満を満たす見込み（満たさなければ pair dict の構築を遅延化する。ただし `lru_cache` シングルトンなのでプロセスあたり 1 回）。

**API**（task4 はこれだけを使う）

```python
@dataclass(frozen=True, slots=True)
class CorpusFrequencyTable:
    total_tokens: int                      # property ではなくフィールドで公開（frozen なので等価）
    def pair_count(self, reading: str, surface: str) -> int      # 未観測は 0
    def reading_count(self, reading: str) -> int                 # 未観測は 0
    def surface_count(self, surface: str) -> int                 # 未観測は 0
    @classmethod
    def load(cls, path: str | Path = DEFAULT_CORPUS_FREQ_PATH) -> "CorpusFrequencyTable"
    @classmethod
    def from_counts(cls, pairs: Mapping[tuple[str, str], int],
                    surfaces: Mapping[str, int], total: int) -> "CorpusFrequencyTable"
    def dump(self, path: str | Path) -> str                      # 書いた表の SHA-256 を返す
```

- `load()` は **`path.exists()` を先に見て、無ければ明示的に `FileNotFoundError`** を投げる（メッセージに再生成コマンド `scripts/build_corpus_freq.py` を書く）。黙って空表を返す経路をコード上に作らない。
- `from_counts` は `pairs` から `reading_count` を導出し、`surfaces` は**引数のものをそのまま保持する**（導出しない）。§2.2 の数え方では `c(s) == Σ_r c(r,s)` かつ `N == Σ c(r,s)` が恒等的に成立するが、API を導出に潰すと将来「読みが取れないトークンも表記だけ数える」拡張ができなくなる。恒等性は**テストと manifest の invariant チェックで担保**する（§2.3 `counts.marginals_consistent`）。
- `dump()` は親ディレクトリを作らない（呼び出し側の責任）。返り値の SHA-256 を manifest にそのまま入れることで、「書いたバイト列」と「manifest の記録」が別経路で食い違う余地を消す。

### 2.2 `scripts/build_corpus_freq.py`

**import**（写経禁止の遵守）

```python
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))   # 先例: scripts/evaluate_d1.py:31
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from build_d1_data import (
    article_split, node_reading, iter_hf_articles, iter_jsonl_articles,
    load_exclusion_terms, overlaps_evaluation,
    DEFAULT_DATASET, DEFAULT_CONFIG, DEFAULT_REVISION,
)
from beme.corpus_freq import CorpusFrequencyTable
```

`article_split` のハッシュ式・`_SPLIT_RANGES`・`node_reading` のフィールド添字（`fields[7]`）を**このファイルに一切書かない**。定数（dataset/config/revision）も `build_d1_data` から import して既定値にする（`data/d1/manifest.json` の値と一致することを起動時に assert する。§2.5）。

**CLI**（指示書の 8 個 + 追加 4 個。追加はすべて理由付き）

```
--output PATH            既定 data/corpus_freq_v1.bin
--manifest PATH          既定 data/corpus_freq_v1.manifest.json
--dataset / --config / --revision   既定 = build_d1_data の DEFAULT_*
--target-bytes INT       既定 104857600
--max-articles INT       スモーク用
--split-seed INT         既定 20260822
--articles-jsonl PATH    ローカル記事ソース（テスト用）
--d1-dir PATH            既定 data/d1
--coverage-pairs PATH    既定 artifacts/issue19/s-measurement/coverage_pairs.jsonl（★追加・照合専用）
--allow-missing-d1       ★追加。d1 が無い環境で監査を明示的に放棄する。既定 false
--progress-every INT     既定 100。build_d1_data:459 と同じ stderr 進捗
--json                   manifest を stdout に出す（build_d1_data:602 と同じ）
```

`--coverage-pairs` は §0 #3 の照合に使う（判定値の算出には使わない）。`--allow-missing-d1` は「d1 が無いと監査が黙ってスキップされる」という**リーク監査を無価値にする最悪の失敗モード**を塞ぐためのもの。既定では d1 が無ければ `leak_audit.passed=false` を書いて**非ゼロ終了**する。

**処理フロー**

1. `GenericTagger(f"-r /dev/null -d {ipadic.DICDIR}")` を作る（`build_d1_data:393` と同一。`fugashi` / `ipadic` の ImportError は同じメッセージで `RuntimeError`）
2. `exclusion_terms = load_exclusion_terms()`（実測 **238 件**）
3. 記事ソース: `--articles-jsonl` があれば `iter_jsonl_articles`、無ければ `iter_hf_articles(dataset, config, revision)`
4. 各記事について、**`build_d1_data.build()` の走査窓ロジックを厳密に踏襲する**:
   ```python
   encoded_bytes = len(article.text.encode("utf-8"))
   if bytes_seen and bytes_seen + encoded_bytes > args.target_bytes:
       break                      # 記事は切り詰めない。build_d1_data:408-409 と同一
   bytes_seen += encoded_bytes    # ← スキップする記事も必ずここで数える
   articles_seen += 1
   ```
   `--max-articles` は本体処理の**後**に `articles_seen >= max_articles` で break（走査窓の判定より後ろ）
5. フィルタは **`build_d1_data` と同じ順序**で 2 段（順序は counters の帰属だけに影響し、通過集合は順不同で同一）:
   ```python
   if overlaps_evaluation(article.text, exclusion_terms):
       evaluation_overlap_articles_skipped += 1;  continue     # build_d1_data:412-414 と同順
   if article_split(article.article_id, args.split_seed) != "train":
       dev_test_articles_skipped += 1;  continue
   ```
   この順序を採る理由: d1 manifest の `articles_excluded_for_evaluation_overlap = 3` と**直接比較できる**（§2.5 の検証 3）。指示書 §2.2 は split → overlap の順で書いているが、両方適用する要件は満たしたうえで、比較可能性を取る。
6. 通過した記事のみ:
   ```python
   train_articles_counted += 1
   counted_article_ids.add(article.article_id)          # 監査用。100MiB 窓で高々 5,104 件
   for node in tagger(article.text):                    # build_d1_data:147-152 と同じ全文一括
       reading = node_reading(node)
       if reading is None:  continue
       pair_counts[(reading, node.surface)] += 1
       surface_counts[node.surface] += 1
       total_tokens += 1
   ```
7. `CorpusFrequencyTable.from_counts(...)` → `dump(args.output)` → 返り値の SHA-256 を manifest に格納
8. リーク監査（§3）→ 被覆率（§2.4）→ manifest 書き出し → 監査不合格なら**非ゼロ終了**

### 2.3 manifest（`data/corpus_freq_v1.manifest.json`）

指示書 §2.3 のスキーマをそのまま採用し、**検証に必要な 4 キーだけ追加**する（追加はすべて「後から検証できるようにする」ためのもの）。

```jsonc
{
  "schema_version": 1,
  "created_on": "2026-08-28",
  "source": {"dataset": "...", "config": "20231101.ja", "revision": "b04c8d1...",
             "utf8_bytes_scanned": 104852271,
             "byte_accounting": "build_d1_data: every article counted before filtering; break before exceeding target; articles are never truncated"},   // ★追加（指示書 §2.2-4 が要求する記録）
  "split": {"algorithm": "sha256(seed:article_id) modulo 10000", "seed": 20260822, "train_upper": 7700},
  "articles": {"seen": 5104, "train_counted": 0, "dev_test_skipped": 0, "evaluation_overlap_skipped": 3},
  "counts": {"total_tokens": 0, "unique_pairs": 0, "unique_surfaces": 0, "unique_readings": 0,
             "marginals_consistent": true},                                    // ★追加 Σ_r c(r,s)==c(s) かつ Σc(r,s)==N
  "leak_audit": {"d1_dev_test_article_ids": 194, "counted_article_ids_intersecting_d1_dev_test": 0,
                 "d1_dev_test_article_ids_misclassified_as_train": 0,          // ★追加（逆方向の検査）
                 "d1_available": true, "passed": true},                        // ★追加（d1 欠如を pass と誤読させない）
  "coverage": {"d1_test_gold_pairs": 2000, "d1_test_gold_pairs_seen": 0,
               "d1_test_gold_pair_coverage": 0.0,
               "d1_test_gold_surfaces_seen": 0, "d1_test_gold_surface_coverage": 0.0,
               "d1_train_gold_pair_coverage": 0.0,
               "d1_test_gold_pair_coverage_span1_only": 0.0,                   // ★追加（§4.1 の構造的上限の可視化）
               "source": "recomputed from data/d1/{train,test}.jsonl (reading, positive); cross-checked against coverage_pairs.jsonl",
               "coverage_pairs_sha256": "227f75c0...", "coverage_pairs_match": true},
  "table_sha256": "...",
  "environment": {"git_commit": "...", "python": "3.13.5", "host": "..."}      // ★追加（task1 レポート §7 の規約に合わせる）
}
```

`created_on` は d1 と同じく手動固定（実行日）。`ensure_ascii=False, indent=2, sort_keys=True` + 末尾改行（`build_d1_data:557` と同一）。

### 2.4 被覆率の算出（分岐 C の判定値）

`--coverage-pairs` を判定に使わず、**split 別に d1 JSONL から再構成する**（§0 #3）。

```python
# split ごとに (reading, positive) の多重集合を作る
per_split[split] = Counter((row["reading"], row["positive"]) for row in jsonl(d1_dir / f"{split}.jsonl"))
# 出現数重み付き（task1 の count が出現数なので同じ重み付け）
seen = sum(n for (r, s), n in per_split["test"].items() if table.pair_count(r, s) > 0)
d1_test_gold_pair_coverage = seen / sum(per_split["test"].values())
```

- **重み付けは出現数**（ユニークペアではない）。`coverage_pairs.jsonl` の `count` が出現数であり、分母が 12,000 = ケース数と一致することから、task1 の意図は出現数重み付けである。ユニーク基準の値も参考として `unique` サフィックス付きで併記する。
- 照合: `per_split["train"] + per_split["test"]` が `--coverage-pairs` の内容と**完全一致**することを assert し、`coverage_pairs_match` に記録する。一致しなければ非ゼロ終了（task1 の定義から silent に drift していないことの証明）。`artifacts/` は gitignore なのでファイルが無い場合は `coverage_pairs_match: null` + 警告（判定値の算出には影響しない）。
- `d1_test_gold_surface_coverage` は `table.surface_count(positive) > 0` の出現数比。
- `d1_test_gold_pair_coverage_span1_only` は `span_tokens == 1` の 1,730 件に限った被覆率（§4.1）。

### 2.5 `tests/test_corpus_freq.py`

指示書 §2.4 の 5 項目 + 監査の**負のテスト**。`data/d1/*.jsonl` と `data/corpus_freq_v1.bin` に一切触れない（`tmp_path` に合成 d1 を作る）。ビルダーはサブプロセス実行（`subprocess.run([sys.executable, "scripts/build_corpus_freq.py", ...])`）で終了コードまで検証する — task1 の `tests/test_measure_local_ceiling.py` と同じ流儀。

| # | テスト名 | 内容 |
|---|---|---|
| 1 | `test_table_round_trip_is_exact` | `from_counts` → `dump` → `load` で 3 メソッド + `total_tokens` が完全一致 |
| 2 | `test_dump_is_deterministic` | 挿入順を変えた 2 つの `dict` から作った表の `dump` の SHA-256 が一致 |
| 3 | `test_load_missing_path_raises` | 存在しないパスで `FileNotFoundError`（空表を返さないこと・メッセージに再生成手順） |
| 4 | `test_truncated_table_raises` | ヘッダ/ペイロードを切ったファイルで `ValueError`（`matrix_cost` と同じ堅牢性） |
| 5 | **`test_only_train_bucket_articles_are_counted`** | `article_split` を**総当たりで呼んで** train / dev / test 各バケットに落ちる `article_id` を 1 つずつ見つけ（ハッシュ値をハードコードしない）、3 記事の `--articles-jsonl` を作る。train 記事にのみ含まれる表記のトークンが `pair_count > 0`、dev/test 記事にのみ含まれる表記が **`pair_count == 0`**。manifest の `dev_test_skipped == 2` |
| 6 | **`test_evaluation_overlap_article_is_excluded`** | `load_exclusion_terms()` の実値を 1 つ取り、**train バケットに落ちる** article_id の記事本文に埋める。その記事の固有表記が数えられておらず `evaluation_overlap_skipped == 1` であること（split ではなく overlap で落ちたことの証明） |
| 7 | **`test_leak_audit_fails_on_dirty_d1`** | `tmp_path` の合成 `dev.jsonl` に**train バケットの article_id** を入れて `--d1-dir` に渡す → 終了コード非ゼロ・`leak_audit.passed == false`。監査が実際に落ちる経路を持つことの証明 |
| 8 | `test_missing_d1_is_not_a_silent_pass` | `--d1-dir` が空ディレクトリ → 非ゼロ終了。`--allow-missing-d1` を付けると 0 終了かつ `d1_available == false` / `passed == false` |
| 9 | `test_marginals_are_consistent` | 合成ビルド結果で `Σ_r c(r,s) == c(s)` かつ `Σ c(r,s) == total_tokens` |

新規 **9 件**を見込む（→ 完了条件 654 passed / 2 failed。§5）。テスト 5〜9 は `fugashi` / `ipadic` を要する（dev group に既存）ので、`pytest.importorskip("fugashi")` は**付けない**（ローカルの合否をぼかすため）。CI 側で問題になれば task7 で申し送る。

---

## 3. リーク対策 3 重の具体的適用

PLAN §3.4 の 3 層を、コード上のどこで担保するかを明記する。

**第 1 層 — 記事 split フィルタ（写経の排除）**

`from build_d1_data import article_split` のみ。`build_corpus_freq.py` に `hashlib.sha256(f"{seed}:{article_id}"...)` という文字列が**一度も現れない**ことを、レビュー時に `grep -n "sha256\|10_000\|7700" scripts/build_corpus_freq.py` で確認する（ヒットは表の SHA-256 計算箇所のみであるべき）。`_SPLIT_RANGES` / `train_upper=7700` も manifest に書くときは import 側から引く（`build_d1_data._SPLIT_RANGES[0][1]`）。

**第 2 層 — 評価スイート除外**

`load_exclusion_terms()` / `overlaps_evaluation()` をそのまま呼ぶ（§2.2 手順 5）。落とすと d1 は汚さずに AJIMEE / handmade を汚す。238 term × 5,104 記事の部分文字列探索は d1 ビルドで実績があり、コスト上の懸念は無い。テスト 6 がこの層の生存確認。

**第 3 層 — 監査 assert と非ゼロ終了**

ビルダー末尾で、**数えた記事集合の側から**独立に検査する:

```python
d1_ids = {json.loads(l)["article_id"] for name in ("dev", "test")
          for l in (d1_dir / f"{name}.jsonl").read_text().splitlines()}
intersection = counted_article_ids & d1_ids                      # 0 でなければならない
misclassified = [a for a in d1_ids if article_split(a, seed) == "train"]   # 逆方向。0 でなければならない
passed = (not intersection) and (not misclassified) and d1_available
manifest["leak_audit"] = {...}
write_manifest()            # ← 落ちる場合でも manifest は書く（証拠を残す）
if not passed:
    print(f"LEAK AUDIT FAILED: intersection={len(intersection)} misclassified={len(misclassified)}", file=sys.stderr)
    return 1                # 非ゼロ終了。表は残すが manifest に passed=false が残る
```

2 方向の検査を両方持つ理由: `intersection` は「数えてはいけない記事を数えた」を、`misclassified` は「seed / 分割式そのものがズレている」を捉える。前者だけだと、seed を間違えた場合に「d1 の dev/test が train 判定される → そもそも intersection が発生しない」形で**すり抜ける**。d1 の dev/test 記事は実測 **194 件**。

---

## 4. 工程と完了基準

### 工程 0 — 実装（コード 3 本 + .gitignore）

完了基準: `uv run pytest -q tests/test_corpus_freq.py -v` が新規 9 件すべて green。`uv run ruff`（設定があれば）/ `git diff --check` が無出力。

### 工程 1 — スモーク（記事 200 本）

```bash
mkdir -p /tmp/claude-1000
uv run python scripts/build_corpus_freq.py --max-articles 200 \
  --output /tmp/claude-1000/cf-smoke.bin --manifest /tmp/claude-1000/cf-smoke.json
```

完了基準:
- 終了コード 0・`leak_audit.passed == true`・`counted_article_ids_intersecting_d1_dev_test == 0`
- `articles.seen == 200`、`train_counted + dev_test_skipped + evaluation_overlap_skipped == 200`
- `train_counted / 200` が **0.74〜0.80** に入る（train 域は 77%。ここが 1.0 なら split フィルタが効いていない = 最も危険な失敗を最速で検出する）
- `counts.marginals_consistent == true`・`total_tokens > 0`
- 同じコマンドを 2 回走らせて `table_sha256` が一致（決定性の実地確認）
- **所要時間を記録し、5,104 記事へ線形外挿する**（工程 2 の見積り根拠）

### 工程 2 — 本番ビルド（d1 と同じ 100MiB 窓）

```bash
uv run python scripts/build_corpus_freq.py \
  --output data/corpus_freq_v1.bin --manifest data/corpus_freq_v1.manifest.json
```

完了基準（**走査窓が d1 と一致していることを数値で証明する**。§2.2 の byte accounting を厳密に踏襲した結果として、以下は一致するはずである）:

| 項目 | 期待値 | 出所 |
|---|---|---|
| `source.utf8_bytes_scanned` | **104,852,271** | `data/d1/manifest.json:source.source_utf8_bytes_scanned` |
| `articles.seen` | **5,104** | 同 `articles_processed` |
| `articles.evaluation_overlap_skipped` | **3** | 同 `articles_excluded_for_evaluation_overlap` |
| `articles.train_counted` | 約 3,900（5,104 − 3 の 77%） | — |
| `leak_audit.passed` | `true`・交差 0・誤分類 0 | §3 |
| `counts.unique_pairs` | 20〜50 万（指示書の 34 万想定と同オーダー） | — |

上 3 つが一致しなければ**走査窓が d1 とズレている**ので、`coverage` の解釈が変わる。不一致のまま先に進まない（原因を特定して再ビルド）。HF の revision がピン留めされている（`b04c8d1c…`）ので、ストリーム順は再現するはずである。

### 工程 3 — 検証コマンド

指示書 §5 をそのまま実行（パスは `/home/sarashin/beme-19`）。加えて:

```bash
# 1) 表が実際に読めて速いこと
uv run python -c "
import time; from beme.corpus_freq import CorpusFrequencyTable
t0=time.perf_counter(); tbl=CorpusFrequencyTable.load(); dt=time.perf_counter()-t0
print('load sec', round(dt,3), 'N', tbl.total_tokens)
assert dt < 1.0"

# 2) 表の SHA-256 と manifest の一致
sha256sum data/corpus_freq_v1.bin
uv run python -c "import json;print(json.load(open('data/corpus_freq_v1.manifest.json'))['table_sha256'])"

# 3) 走査窓が d1 と一致していること
uv run python -c "
import json
d1=json.load(open('data/d1/manifest.json'))['source']; cf=json.load(open('data/corpus_freq_v1.manifest.json'))
assert cf['source']['utf8_bytes_scanned']==d1['source_utf8_bytes_scanned'], (cf['source']['utf8_bytes_scanned'], d1['source_utf8_bytes_scanned'])
assert cf['articles']['seen']==d1['articles_processed']
assert cf['articles']['evaluation_overlap_skipped']==d1['articles_excluded_for_evaluation_overlap']
print('scan window matches d1:', cf['source']['utf8_bytes_scanned'])"

# 4) 探索コアと特徴が無変更
git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py src/beme/features.py   # 空
git diff --check && git status --short
```

### 工程 4 — 分岐 C の判定（§5）と報告

完了基準: `coverage.d1_test_gold_pair_coverage` の実測値と判定、および `leak_audit` の全値を報告冒頭に書く。

---

## 5. 分岐 C の判定と対応

判定値 = `coverage.d1_test_gold_pair_coverage`（d1 **test** の gold (読み,表記) ペア・**出現数重み付け**・分母 2,000）。閾値は事後に動かさない。

| 実測値 | 対応 |
|---|---|
| **≥ 0.70** | そのまま task4 へ。追加ビルド無し |
| **0.50 〜 0.70** | `--target-bytes 419430400`（4 倍）で `data/corpus_freq_v1.bin` を**上書き再ビルド**し再測。改善しなければ task4 へ「4 特徴目 `lex.corpus_surface_unseen` を足す」と申し送る |
| **< 0.50** | 4 倍再ビルド **かつ** 4 特徴目の申し送りを**両方**行う。再ビルド後も < 0.50 なら task7 へ「F2 の効く範囲は d1 の半分未満」と記録する申し送りを書く |

いずれでも **task4 は実行する**（PLAN §3.5 末尾）。

4 倍再ビルド時の扱い:
- 出力は同じ `data/corpus_freq_v1.bin` / `data/corpus_freq_v1.manifest.json` を上書きする（task4 が既定パスから読むため）。**100MiB 版の manifest は `data/corpus_freq_v1.manifest.100mib.json` として退避し、報告に両方の被覆率を並べる**（閾値判定の履歴を消さないため）
- 4 倍窓は d1 の走査窓と**一致しなくなる**（工程 2 の 3 検証は当然不一致になる）。これは意図した逸脱であり、manifest の `utf8_bytes_scanned` が 419,430,400 前後であることを確認すればよい。**リーク対策 3 層は窓の大きさと独立に効く**（split フィルタは記事 id ベースであり、窓を広げても dev/test 記事が入り込む余地は無い）
- 所要時間は工程 2 の約 4 倍。実行前に見積りを報告する

### 5.1 判定前に記録しておく構造的上限（重要）

コーパスの `c(r,s)` は **MeCab ノード単位**で数えるが、d1 の gold ペアは**ケースのスパン単位**（`(reading, positive)`）である。d1 test の `span_tokens` 内訳は **1: 1,730 / 2: 143 / 3: 127**（実測）。span が 2 以上の 270 件（**13.5%**）は、連結後の (読み, 表記) が単一 MeCab ノードとして現れることが実質的に無いため、コーパスをどれだけ広げても `c(r,s) > 0` になりにくい。

したがって **`d1_test_gold_pair_coverage` の実質的な上限は約 0.865** である。閾値 0.70 はこの上限の内側なので分岐 C は有効に機能するが、判定値が 0.70 を割ったときに「コーパスが足りない」のか「多トークンスパンの構造的な取りこぼし」なのかを分離できないと、4 倍再ビルドの是非を誤る。よって:

- `d1_test_gold_pair_coverage_span1_only`（分母 1,730）を manifest に併記し、**判定の解釈に必ず添える**
- **判定そのものは指示書どおり全件（分母 2,000）の値で行う**。閾値を上限に合わせて割り引くような事後調整はしない
- この構造的上限は task4 の 3 特徴の実効性とは別問題である点も報告に書く（特徴は lattice のトークン単位で評価されるので、多トークンパスでも各トークンが個別に `c(r,s)` を引ける。被覆率の低さがそのまま特徴の無効化を意味しない）

---

## 6. 完了条件（チェックリスト）

- [ ] `src/beme/corpus_freq.py` が §2.1 の 7 API を持ち、決定的に dump/load できる（load < 1 秒）
- [ ] `scripts/build_corpus_freq.py` が §2.2 の CLI で動く
- [ ] `tests/test_corpus_freq.py` が §2.5 の 9 項目（**リーク検査と監査の負のテストを含む**）
- [ ] `uv run pytest -q` が **`645 + 新規 9 = 654 passed / 2 failed`**（失敗は `tests/test_audit_leakage.py` の 2 件のみ）
      ※ 指示書の「638 + 新規」は task1 着手前の値。現在の実測ベースラインは 647 collected = 645 passed / 2 failed（§0 #2）
- [ ] `data/corpus_freq_v1.bin` 生成済み・`data/corpus_freq_v1.manifest.json` の `leak_audit.passed == true`
- [ ] `counted_article_ids_intersecting_d1_dev_test == 0` かつ `d1_dev_test_article_ids_misclassified_as_train == 0`
- [ ] 走査窓の 3 値が d1 manifest と一致（`104852271` / `5104` / `3`）
- [ ] `coverage.d1_test_gold_pair_coverage` が埋まり、分岐 C の判定が報告に書かれている（`_span1_only` 併記）
- [ ] `coverage_pairs_match == true`（task1 の定義から drift していない）
- [ ] `sha256sum data/corpus_freq_v1.bin` == manifest の `table_sha256`
- [ ] `.gitignore` に `data/corpus_freq_v1.bin` が 1 行追加されている（manifest は ignore されていないことを `git check-ignore -v data/corpus_freq_v1.manifest.json` が**何も返さない**ことで確認）
- [ ] `git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py src/beme/features.py` が**空**
- [ ] `git diff --check` / `git status --short` を確認済み、**`git commit` / `git push` をしていない**

---

## 7. 制約（厳守）

- `src/beme/lattice.py` / `src/beme/converter.py` **変更禁止**（探索コア）
- `src/beme/features.py` も**このタスクでは変更しない**（3 特徴の実装は task4）
- **`article_split` のハッシュ式を写経しない。** `build_d1_data` から import する。`node_reading` / `overlaps_evaluation` / `load_exclusion_terms` / `iter_*_articles` も同様
- **`overlaps_evaluation` を落とさない**（落とすと AJIMEE / handmade を汚す）
- 依存パッケージを増やさない（標準ライブラリ + 既存 `dependency-groups.dev` の `datasets` / `fugashi` / `ipadic` / `pytest` のみ）
- `git commit` / `git push` をしない。`git stash` を使わない。作業は `/home/sarashin/beme-19` の中だけ
- `tests/test_audit_leakage.py` の既存 2 件の失敗は**直さない**（環境固有。PLAN §2.7）。それ以外の 645 件は 1 件も落とさない
- 表本体は gitignore、manifest はコミット対象（commit 自体は行わない）
- 分岐 C の閾値（0.70 / 0.50）を事後に動かさない
- 報告は日本語・数値と結論を先頭に・`leak_audit` と `coverage` の実測値を必ず含める

---

## 8. リスクと注意点

| # | リスク | 影響 | 対策 |
|---|---|---|---|
| 1 | **HF ストリーミングの時間とネットワーク** | 100MiB 相当（5,104 記事）は `20231101.ja` の先頭シャードに収まる見込みだが、初回はシャードのダウンロードが走る。数分〜十数分 | 工程 1 で 200 記事の所要時間を測り線形外挿。`--progress-every 100` で stderr に進捗を出す。中断されたら最初からやり直し（**チェックポイントは作らない** — 部分集計は決定性と監査を壊す） |
| 2 | **MeCab 解析コスト** | train 域 ~77% × 100MiB ≒ 80MiB を全文タグ付け。Python 側のノード反復が支配的で、数分〜十数分 | 工程 1 で実測。`article_surface_frequencies`（`build_d1_data:147`）と同じ全文一括方式なので d1 ビルドの実績内 |
| 3 | **走査バイト数の数え方のズレ** | ズレると `coverage` が d1 と別の窓の話になり、分岐 C の判定が無効 | `build_d1_data.build()` のループを厳密に踏襲（切り詰めない・スキップ記事も数える・超過前に break）。工程 2 で `104852271` / `5104` / `3` の 3 点一致を**必須の完了基準**にした。指示書は「fetch_corpus.py と同じ数え方」と書くが、`fetch_corpus.py:84` は最後の記事を**バイト単位で切り詰める**ため d1 の窓とは一致しない。要件（d1 と窓を揃える）を満たすのは build_d1_data 側の規則であり、そちらを採る。採用した規則は manifest の `source.byte_accounting` に明記する |
| 4 | **coverage_pairs.jsonl に split が無い**（§0 #3） | manifest の必須キーが埋まらない / 誤った値が入る | d1 JSONL から split 別に再構成し、マージ結果が `coverage_pairs.jsonl` と完全一致することを assert（照合済み: 10,342 ユニーク・12,000 出現で一致） |
| 5 | **多トークンスパンによる被覆率の構造的上限 ≈ 0.865** | 分岐 C の判定を「コーパス不足」と誤診し、無駄な 4 倍再ビルドを走らせる | `_span1_only` を併記して解釈材料にする。ただし**判定は全件の値で行う**（閾値の事後調整はしない）。§5.1 |
| 6 | **d1 が無い環境で監査が空回りする** | リーク監査が「実行されていないのに passed」になる最悪の失敗 | 既定で d1 欠如は非ゼロ終了。`--allow-missing-d1` を明示したときのみ続行し、`d1_available: false` / `passed: false` を残す。テスト 8 で担保 |
| 7 | **`counted_article_ids` のメモリ** | 100MiB 窓で 5,104 件・4 倍でも 2 万件程度 | そのまま set で保持（問題にならない）。ただし将来さらに窓を広げる場合は article_id をソート済みファイルに書き出す方式へ |
| 8 | **`data/d1/*.jsonl` が symlink**（`/home/sarashin/beme-d1/repo/` 参照） | リンク先が消えると監査と被覆率が走らない | 工程 2 の前に `wc -l data/d1/{train,dev,test}.jsonl` で 10,000 / 1,000 / 2,000 を確認する |
| 9 | **同一文の記事間重複**（PLAN §3.4 の残存リスク） | article_id ベースでは防げないコピペ節による軽微な汚染 | 追わない。**既知の限界として報告に明記する**（PLAN が既にブロッカーでないと判断済み） |
| 10 | **`from_counts` の `surfaces` 冗長性** | v1 では `c(s) == Σ_r c(r,s)` が恒等的に成立し、引数が実質無意味 | API は指示書どおり維持し、恒等性を `counts.marginals_consistent` とテスト 9 で検査する。導出に潰して将来の拡張余地を消さない |
| 11 | **`GenericTagger` の生成コスト / ImportError** | dev group 未同期の環境で不明瞭な失敗 | `build_d1_data:380-384` と同じ `RuntimeError("fugashi and ipadic are required; run 'uv sync --group dev'")` を踏襲 |
| 12 | **`datasets` のストリーミングスレッドが終了時に残る** | プロセスが終わらない | `build_d1_data:606-611` と同じ `os._exit(status)` パターンを採用（stdout/stderr を flush してから） |

---

## 9. 見積り

| 工程 | 内容 | 見込み |
|---|---|---|
| 0 | 実装 3 本 + テスト 9 件 | 中 |
| 1 | スモーク 200 記事（2 回・決定性確認込み） | 数分 |
| 2 | 本番 100MiB ビルド | 10〜30 分（工程 1 の実測で確定） |
| 3 | 検証コマンド一式 + 全体 pytest | 約 3 分 |
| 4 | 分岐 C 判定・報告 | 短 |
| (条件付き) | 4 倍再ビルド | 工程 2 の約 4 倍 |

---

**この計画で着手してよいか、特に §0 の 3 点（worktree パス・pytest ベースライン 645・coverage_pairs.jsonl の split 欠如への対処）と §5.1（被覆率の構造的上限 0.865 を判定に反映せず併記のみとする方針）について確認をお願いします。**
