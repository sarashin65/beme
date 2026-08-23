# 最初の worker（opencode luna）への指示文案

以下をそのまま貼る。PLAN §0 と同一内容に、完了報告の要求を明示的に足したものである。

---

Issue #9 Phase 4 のタスク 1 を実施してください。

リポジトリ: `/home/sarashin/work/beme-9p4`（ブランチ `feat/issue9-phase4`・クリーン）

まず次を順に読んでください。

1. `docs/missions/issue9-phase4/PLAN.md` の §1〜§2.1 と §3.1（Phase 4 の前提と、なぜ Top-1 の定義統一が最優先なのか）
2. `docs/missions/issue9-phase4/task1-mission.md`（あなたの指示書。目的・対象ファイル・変更内容・完了条件・検証コマンド・制約がすべて書いてあります）
3. `src/beme/training.py`（`evaluate_split`）・`src/beme/lattice.py`（`k_best_viterbi` と `_paths_by_node` の `_path_key`）・`src/beme/converter.py`（`convert` の dedup と `(cost, surface)` ソート）

タスク 1 は「Top-1 の入口の定義を 1 つに統一する」実装です。Phase 4 は全部が「リランカ前後の Top-1 差」の測定なので、入口が 2 種類あるまま先に進むと Phase 3 と同じ罠（A3 で 208 件の乖離）を踏みます。

**`k_best_viterbi` のヒープキーは変えないでください。** ヒープは A\* の f 値 `(cost + after_edge, cost, sequence, position, path)` で pop しており、`sequence` は単なる挿入順カウンタです。表層は終端に着くまで確定しないので、探索中に表層でタイブレークすることは原理的にできません。統一は「同点の最小コスト経路群を列挙してから `(cost, surface)` の最小を取る」共有ヘルパで行ってください。設計は PLAN §3.1 の案 C にあります。

落とし穴を 2 つ先に伝えます。

- `evaluate_split` は学習ループの内側で train 10,000 + dev 1,000 に対して毎 epoch 走ります。`beam_width=1` を `64` にすると学習時間が跳ねます。同点が無い限り 1 回目の pop で確定する early-stop を必ず入れて、**1 epoch の wall time が現行比 ×1.2 以内**であることを実測で示してください。
- `Converter.convert` は dedup 後に `(cost, surface)` でソートします（`converter.py:169`）。`_paths_by_node` の `_path_key` は `(cost, surface, tokens)` で既に表層を見ていますが、`k_best_viterbi` は見ていません。**直すのは `k_best_viterbi` を使う側であって `_path_key` ではありません。**

既存の 12 条件の数値のうち、同点が発生しない条件（A0 を含む 11 条件）は 1 bit も変わらないはずです。`uv run pytest -q` は現在 585 passed です。

`git commit` / `git push` はしないでください。変更は作業ツリーに残してください。

---

## 完了報告に含めるもの（**実出力を貼ること。要約や「通りました」は不可**）

次のコマンドを実行し、**標準出力をそのまま貼り付けて**報告してください。

```bash
# 1. 新規テスト
uv run pytest -q tests/test_top1_unification.py

# 2. 全テスト（現在 585 passed。増えているはず）
uv run pytest -q

# 3. 不変条件: Phase 2/3 と bit 一致すること
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json --json

# 4. 1-best を k_best_viterbi から直接取っている箇所の棚卸し
grep -rn "k_best_viterbi" src/ scripts/

# 5. 作業ツリーの状態
git status --short
git diff --stat
```

コマンド 3 の期待値（Phase 2 / Phase 3 で固定済み。**1 桁でも違ったら報告して止まること**）:

```text
Handmade: NDCG@5 0.992972 / MRR@5 1.000000 / Top-1 1.000000
AJIMEE:   CER 0.290080 / exact 0.070000
Mozc:     88/498
```

加えて、次を文章で報告してください。

1. **同点テストが修正前のコードで FAIL することの確認結果。** 「先に落ちるテストを書いてから直した」ことの証拠です。実際に修正前へ戻して実行した出力を貼ってください。
2. **1 epoch の wall time**（現行 / 変更後 / 倍率）と、**何のデータで測ったか**。miniPC には `data/d1/*.jsonl` が無いので代替計測になります。代替であることを明記してください。
3. **コマンド 4 の grep 結果のうち、統一の対象外にした箇所とその理由。** 対象外がゼロならそう書いてください。
4. **Phase 3 の A3 条件（`decomposed_v1-no_ipadic_cost`）で乖離が解消したか。** A3 は同点が大量に出る唯一の条件です。手元で再現できるなら確認結果を、再現できないなら「GPU 側の Task 5 で確認する」と書いて先へ進んでください。
5. **指示書の完了条件のうち満たせなかったものがあれば、名指しでその理由。** 満たせないまま「完了」と報告しないでください。
