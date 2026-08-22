# Task 2: IPADIC matrix cost 特徴と遷移状態の拡張

## 目的

`ψ(previous_edge, current_edge)` に IPADIC `matrix.def` 由来の接続コストを実数値特徴 `trans.matrix_cost` として導入する。あわせて、遷移状態を `pos` のみから `(pos, right_id)` へ拡張できるようにする。**既存 spec の遷移状態と数値は 1 bit も変えない。**

最初に `docs/missions/issue9-phase3/PLAN.md` の §2.2(c)・§3.2・§4（M0/M1 行）、`src/beme/features.py`、`src/beme/scorer.py`、`src/beme/lattice.py`（特に `backward_costs_by_state` と `k_best_viterbi`）、`src/beme/oracle.py`、`src/beme/ipadic.py`、`scripts/build_ipadic_dictionary.py` を読むこと。

## 依存

Task 1（特徴仕様レジストリと特徴フィルタ）が完了していること。

## 前提（実測済み・信じてよい）

- `matrix.def` は **miniPC の `/home/sarashin/.cache/beme-diag/ipadic/matrix.def`** にある。1316×1316・1,731,857 行・23,008,378 バイト。1 行目が `1316 1316`、以降 `left right cost`。
- GPU マシンには `matrix.def` が無い。`ipadic` pip パッケージが同梱するのは `matrix.bin`（コンパイル済み）だけで `matrix.def` は入っていない。
- `data/dictionary_v2.tsv` に実在する id は left / right とも **801 個**（最大 1315）。必要なペアは 801×801 = **641,601**。
- 遷移状態を `pos` から `(pos, right_id)` へ広げたときの K-best 実測: Mozc 150 読み（平均 10.7 字）で **×1.27**、31 字級の読み 40 件で **×1.41**。遷移キャッシュは 53,093 → 119,523 エントリ。`backward_costs_by_state`（`lattice.py:341-392`）が各位置に実際に入ってくる状態だけを domain にするため、状態語彙 801 でも爆発しない。

## 対象ファイル

- 変更: `src/beme/features.py`（`FeatureSpec.transition_state`・matrix 特徴の抽出・`extractor_state_key`）
- 変更: `src/beme/scorer.py`（`LinearScorer.state_key`）
- 新規: `src/beme/matrix_cost.py`（テーブルのロードと参照）
- 新規: `scripts/build_matrix_table.py`（`matrix.def` → コンパクトバイナリ）
- 新規: `data/ipadic_matrix_v1.bin`（生成物・約 1.3 MB）
- 変更: `scripts/build_ablation_weights.py`（M0/M1 条件の追加）
- 新規: `tests/test_matrix_cost.py`

`src/beme/lattice.py`・`converter.py`・`oracle.py`・`training.py` は**変更しない**。matrix cost は位置非依存なので `LatticePath` の再設計は不要である。

## 変更内容

### 1. matrix テーブルの抽出（`scripts/build_matrix_table.py`）

- 入力: IPADIC の `matrix.def` へのパス（位置引数）と `--dictionary`（既定 `data/dictionary_v2.tsv`）。
- 辞書に実在する left_id / right_id の和集合（+ 文頭文末の id 0）だけを残す。
- 出力 `data/ipadic_matrix_v1.bin` の形式は自分で決めてよいが、次を満たすこと。
  - 標準ライブラリだけで読める（`numpy` に依存しない。`array` / `struct` で十分）。
  - ヘッダに magic・version・id 数・id リストを持ち、本体は `int16` の密行列。
  - 同じ入力から 2 回生成して byte-identical。
- `--json` で統計（id 数・ペア数・cost の min/max/mean・入力 SHA-256）を出す。
- **このスクリプトは miniPC で 1 回だけ実行する。生成物を repo に置いて GPU マシンへ配る。**

### 2. コスト写像（`src/beme/matrix_cost.py`）

raw な matrix cost（実測範囲は `matrix.def` から確認すること。1 行目のサンプルは `0 0 -434`）を BeME のコストスケールへ写す。

- **`beme.ipadic` の既存の raw→BeME 写像の考え方に合わせること。** 新しいマジックナンバーを増やさない。既存定数で表せないなら、その定数をモジュールの公開定数として 1 箇所に定義し、由来をコメントに書く。
- 負のコストが出る（`matrix.def` に負値がある）。学習重みも負になりうるが、ラティスは DAG で `backward_costs_by_state` は残余コストの厳密値を返すため A* の正しさは保たれる（Phase 2 の設計判断）。**負値を切り捨てない。**
- テーブル未収録の id、または `left_id` / `right_id` が欠損（overlay 由来の 133 行）の場合は特徴を立てない（dict に入れない）。

### 3. 遷移状態の拡張

`FeatureSpec` に `transition_state: tuple[str, ...] = ("pos",)` を追加する。

- `("pos",)` のとき、`LinearScorer.state_key` は**現行どおり `edge.token.pos`（`str`）そのもの**を返す。`NamedTuple` でラップしない。既存 DP の状態分割・`LatticePath.state` の型・`backward_costs_by_pos`（`lattice.py:394`・`str | None` キー）を一切変えないため。
- `("pos", "right_id")` のときだけ、`NamedTuple`（例 `TransitionState(pos: str, right_id: int)`）を返す。

### 4. worker が確実に踏む落とし穴（先に潰すこと）

以下は実際にコードを読んで確認した箇所である。片方だけ直すと**例外も警告も出ないまま**壊れる。

- **`transition_features` の `str(previous)`**（`features.py:153`）— ペア名を `f"trans.pair.{previous}>{current}"` で作っている。状態が `NamedTuple` になると `"trans.pair.TransitionState(pos='noun', right_id=12)>particle"` という名前が生成され、どの宣言済み特徴にも一致せず全部 `trans.default` へ落ちる。`previous.pos` を見るように直すこと。
- **`transition_features` の `previous == current`**（`features.py:158`）— `trans.same_pos` の判定。状態全体を比較しているので `right_id` が違うと同 POS でも一致しなくなる。`previous.pos == current` に直すこと。
- **`features.extractor_state_key`**（`features.py:241-249`）— `LinearScorer.state_key` と**同じ値**を返さねばならない。ずれると `path_features` が再構成する遷移が、学習中に実際に使われた遷移と食い違う。両者を同一の関数から導出するか、全 edge で一致することをテストで固定すること。
- **`LinearScorer._transition_cache` のキー**（`scorer.py:52`・`(previous_state, token)`）— 状態が広がるとキャッシュも広がる（実測 53k → 120k）。`TransitionState` は hashable にすること。位置非依存の前提は matrix cost でも保たれる（id は token の属性で `edge.start/end` を見ない）ので、Phase 1 コメントの警告には抵触しない。
- **`src/beme/oracle.py:39`** — `edge_state = scorer.state_key(edge)` を DP 状態に使っている。ファイルを変更する必要は無いが、状態が広がると DP 状態数も増える。オラクル DP が新 spec でも正しく gold を返すことをテストで確認すること。

### 5. spec の追加

Task 1 のレジストリへ次を追加する。

| spec id | 内容 |
|---|---|
| `decomposed_v1+matrix_v1` | `decomposed_v1` + `trans.matrix_cost`、`transition_state=("pos","right_id")` |
| `decomposed_v1+matrix_v1-pos_pairs` | 上記から `trans.pair.*` 13 個を除去 |

初期重みは `trans.matrix_cost` を **0.0** で始める（学習で決める。設計ノート §8 は「0.2 倍程度」を予測しているが、それは検証対象であって初期値ではない）。

## 必須テスト

- `legacy_v1` / `decomposed_v1` の `state_key` の返り値が `str` のままで、本タスク前と同一。
- `decomposed_v1` の全既存指標が 1 bit も変わらない（既存 golden）。
- `extractor_state_key(extractor, edge) == scorer.state_key(edge)` が、辞書から作った十分な数の edge すべてで成立（両 spec で）。
- `+matrix_v1` の `transition_features` が、同 POS・異 right_id の遷移で `trans.same_pos` を正しく立てる。
- `+matrix_v1` の `trans.pair.*` 名が正しく生成される（`str(NamedTuple)` の混入が無い）。
- `trans.matrix_cost` の値が `matrix.def` の該当エントリと写像を通して一致する（数件を手計算で固定）。
- id 欠損（overlay 由来 token）で特徴が立たない。
- 負の matrix cost を含むラティスで `k_best_viterbi` の結果が、小さなラティスの全探索と一致する。
- `constrained_best_path` が `+matrix_v1` spec で正しい gold を返す。
- `scripts/build_matrix_table.py` の 2 回実行が byte-identical。
- 生成したバイナリを別プロセスでロードして同じ値が引ける。

## 完了条件

- `decomposed_v1` の既存挙動が完全に不変。
- `+matrix_v1` で学習と評価の両方が動く。
- **性能ゲート**: `--suite all` の実行時間が `decomposed_v1` 比 **×2.0 以内**。実測（miniPC・`decomposed_v1`）は 15.92 秒なので、31.8 秒以内。超えた場合は実装を最適化するか、超過値を採否判断の材料として報告すること（勝手に特徴を落とさない）。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
# テーブル生成（miniPC でのみ実行可能）
uv run python scripts/build_matrix_table.py \
  /home/sarashin/.cache/beme-diag/ipadic/matrix.def \
  --dictionary data/dictionary_v2.tsv --json

uv run pytest -q tests/test_matrix_cost.py
uv run pytest -q tests/test_oracle.py tests/test_scorer_golden.py tests/test_decomposed_scorer.py
uv run pytest -q

# 既存挙動の不変（期待値: NDCG@5=0.992972 / CER=0.290080 / Mozc 88/498）
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json

# 性能ゲート（31.8 秒以内）
uv run python scripts/build_ablation_weights.py
/usr/bin/time -f "ELAPSED %e s" uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/ablations/decomposed_v1+matrix_v1.json
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- `src/beme/lattice.py`・`converter.py`・`oracle.py`・`training.py` を変更しない。必要性が判明したら勝手に広げず報告する。
- 位置依存特徴を追加しない（matrix cost は位置非依存であり、これを守れば `LatticePath` の再設計は不要）。
- `_edge_key` / `_path_key` と既存タイブレークを変えない。
- `matrix.def` 全体（23 MB）を repo に入れない。生成した約 1.3 MB のバイナリのみ。
- 辞書の負値制約を緩めない。負値は matrix テーブルと学習重み側だけで扱う。
- 新しいマジックナンバーを導入しない。必要なら公開定数にして由来をコメントに書く。
- テストを実装と同時に追加し、上記コマンドの実結果（特に実測秒数）を報告する。
