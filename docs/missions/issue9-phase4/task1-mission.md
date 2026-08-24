# Task 1: Top-1 定義の統一

## 目的

Phase 4 は全部が「リランカ前後の Top-1 差」の測定である。ところが現在、Top-1 を決める経路が 2 つあり、**最安コストが同点のときに別の表記を選ぶ**。入口が 2 種類あるまま測定を始めると、Phase 3 の A3 で起きた「`end_to_end_errors` 588 と diagnose の誤り 796 が 208 件乖離」を再演する。

この乖離を、**規約を 1 つに統一する**ことで解消する。

最初に `docs/missions/issue9-phase4/PLAN.md` の §2.1 と §3.1、`src/beme/training.py`（`evaluate_split`）、`src/beme/lattice.py`（`k_best_viterbi` と `_paths_by_node` の `_path_key`）、`src/beme/converter.py`（`convert` の dedup と `(cost, surface)` ソート）を読むこと。

## 依存

なし。Task 3・4 と並行して進めてよい。Task 2 はこのタスクに依存する。

## 現状（コードで確認済み・推測ではない）

| 経路 | タイブレーク | 該当行 |
|---|---|---|
| `evaluate_split` → `k_best_viterbi(beam_width=1)` | ヒープの挿入順（`sequence` カウンタ） | `training.py:100` / `lattice.py:466-469,494-503` |
| `Converter.convert` → dedup 後ソート | `(cost, surface)` | `converter.py:169` |
| `_paths_by_node` の `_path_key` | `(cost, surface, tokens)` | `lattice.py:413-418` |

**統一先は `(cost, surface)` の最小（= converter の規約）。** 成果物（`diagnose_cases.py`・`oracle_recall.py`・`evaluate.py`）がすべて converter を通るため。`_paths_by_node` は既にこの規約なので、外れているのは `k_best_viterbi` を使う側だけである。

## やってはいけないこと

**`k_best_viterbi` のヒープキーを変えないこと。** ヒープは A\* の f 値 `(cost + after_edge, cost, sequence, position, path)` で pop している。表層は終端ノードに着くまで確定しないので、部分経路の表層で並べると探索の最適性が壊れる。`_path_key` も変えないこと（既に正しい）。

## 対象ファイル

- 変更: `src/beme/training.py`（`evaluate_split` の 1-best 取得）
- 新規または変更: 共有ヘルパの置き場所。`src/beme/lattice.py` に `top1_surface()` を追加するのが素直（`k_best_viterbi` の隣）。`training.py` に置いてもよいが、**Task 2 のハーネスからも import できる場所**にすること
- 新規: `tests/test_top1_unification.py`

`src/beme/converter.py` は変更しない。既に正しい規約である。

## 変更内容

### 1. 共有ヘルパ

```text
def top1_surface(lattice, scorer=None, *, tie_probe=8) -> str | None
```

契約:

- 終端まで到達する経路のうち、**`(cost, surface)` が最小のものの surface** を返す。
- 経路が 1 本も無ければ `None`。
- `Converter.convert(reading, k=1).candidates[0].surface` と**常に一致する**（同じ scorer・同じ `beam_width` のとき）。

実装方針（PLAN §3.1 の案 C）:

```text
paths = k_best_viterbi(lattice, beam_width=tie_probe, scorer=scorer)
if not paths: return None
best = paths[0].cost
tied = [p for p in paths if p.cost == best]
if len(tied) == len(paths):
    # 全部同点 = 打ち切られて同点をまだ見切れていない可能性がある。
    # tie_probe を拡張して取り直す（上限を設けること）
    ...
return min(p.surface for p in tied)
```

- コスト比較は `==` の**厳密一致**でよい。converter 側のタプルソートも float の厳密比較であり、**同じ規約を再現することが目的**である。許容誤差を入れると converter と再び乖離する。
- `tie_probe` の拡張には上限を設け、上限に達したら例外ではなく「その時点の同点群の最小表層」を返してよい。ただし**上限に達した回数を呼び出し側が観測できるようにする**（戻り値を増やすか、カウンタを持たせるか。設計は任せるが、黙って隠さないこと）。

### 2. `evaluate_split` の差し替え

`training.py:100-101` の

```python
prediction_paths = k_best_viterbi(lattice, beam_width=1, scorer=scorer)
prediction = prediction_paths[0] if prediction_paths else None
```

を `top1_surface` 経由にし、`:106` の比較を surface 同士にする。`SplitEvaluation` の形は変えない。

### 3. 性能

`evaluate_split` は学習ループの内側で train 10,000 + dev 1,000 に対して毎 epoch 走る。**同点が発生しない通常モデルでは 1 回目の `k_best_viterbi` 呼び出しで確定する**ように early-stop を設計すること。`tie_probe=8` は `beam_width=1` より重いので、初期値を下げる余地も検討してよい。

**性能ゲート**: `scripts/train_scorer.py` の 1 epoch の wall time が現行比 **×1.2 以内**。実測値を報告に含めること。miniPC には `data/d1/*.jsonl` が無いので、リポジトリ内の小さな合成 split（Task で作る fixture）か、`--suite` 相当の手持ちデータで代替計測してよい。**代替計測であることを報告に明記すること。**

## 必須テスト

1. **同点を意図的に作った小さな合成ラティス**で、`top1_surface` と `Converter.convert(k=1)` が一致する。**このテストは修正前のコードで FAIL すること**を確認してから修正する。
2. 同点が無い通常のラティスで、`top1_surface` が `k_best_viterbi(beam_width=1)[0].surface` と一致する（既存挙動の不変）。
3. 経路が存在しないラティスで `None`。
4. 3 本以上が同点のとき、辞書式最小の表層が選ばれる。
5. `tie_probe` の拡張が発火するケース（同点数 > 初期 `tie_probe`）で正しい答えが出る。
6. `evaluate_split` が `Converter` 経由の Top-1 集計と、同点を含むデータで一致する。
7. 既存の 585 テストが全通し。

## 完了条件

- Top-1 を決める規約がリポジトリ内で 1 つになっている（`k_best_viterbi` の直接利用で 1-best を決めている箇所が `src/beme/` と `scripts/` に残っていないことを grep で確認し、残す場合は理由を報告する）。
- 同点が発生しない条件では既存の数値が 1 bit も変わらない。
- 性能ゲート ×1.2 以内の実測値がある。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_top1_unification.py
uv run pytest -q

# 不変条件: Phase 3 と bit 一致すること
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json --json

# 期待値（Phase 2/3 で固定済み）
#   Handmade NDCG@5 0.992972 / MRR@5 1.000000 / Top-1 1.000000
#   AJIMEE CER 0.290080 / exact 0.070000
#   Mozc 88/498

# 1-best を k_best_viterbi から直接取っている箇所の棚卸し
grep -rn "k_best_viterbi" src/ scripts/

# 性能ゲート（代替計測でよい。何で測ったか報告すること）
```

## 報告に含めること

- 上記コマンドの**実出力**（要約ではなく貼り付け）。
- 同点テストが修正前のコードで FAIL することの確認結果。
- 1 epoch の wall time（現行 / 変更後 / 倍率）と、何のデータで測ったか。
- `grep -rn "k_best_viterbi" src/ scripts/` の結果と、統一の対象外にした箇所があればその理由。
- Phase 3 の A3 条件（`decomposed_v1-no_ipadic_cost`）は同点が大量に出る唯一の条件なので、**手元で再現できるなら**乖離が解消することを確認する。再現できない場合は「GPU 側の Task 5 で確認する」と書いて先へ進んでよい。

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- `k_best_viterbi` のヒープキー・`_path_key`・`Converter.convert` を変更しない。
- `data/d1/*.jsonl` を生成しない・推測で合成しない。テストはリポジトリ内の小さな合成 fixture で完結させる。
- ネットワーク・GPU・外部モデルを必要としないテストにする。
- テストを実装と同時に追加する。
