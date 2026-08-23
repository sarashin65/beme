# Task 1: 特徴仕様レジストリと特徴フィルタ（アブレーション機構）

## 目的

`FeatureSpec` に宣言されていない特徴名を `FeatureExtractor` が出力しないようにし、アブレーション条件を「重みのゼロ化」ではなく「特徴仕様の切り替え」で表現できるようにする。あわせて、spec id から `FeatureSpec` と `FeatureExtractor` を引く単一のレジストリを作り、現在 3 箇所に散っている分岐をそこへ集約する。

最初に `docs/missions/issue9-phase3/PLAN.md` の §2.1・§3.1・§4、`src/beme/features.py`、`src/beme/weights.py`、`src/beme/scorer.py`、`scripts/train_scorer.py`、`scripts/evaluate.py` を読むこと。

## 背景（この設計にした理由）

初期重みで対象特徴を 0 にする方式は使えない。perceptron が 1 回目の更新でその 0 から動かしてしまうため、凍結ロジックを `src/beme/training.py` の更新式へ入れる必要が生じ、Phase 2 が単体テストで固定した符号規約の中核を触ることになる。

一方、`path_features` は `extractor.lexical_features()` / `transition_features()` の返り値だけを合算する（`src/beme/features.py:202-209`）。したがって extractor が出力しない特徴は `feature_delta` にも現れず、学習にもスコアリングにも同時に効く。凍結ロジックが要らない。

さらに `Weights.load` は strict モードで、spec に宣言されていない特徴名を持つ重みファイルを拒否する（`src/beme/weights.py:112-115`）。条件と成果物の取り違えがエラーになる。

## 対象ファイル

- 変更: `src/beme/features.py`
- 変更: `src/beme/weights.py`（spec 解決をレジストリ経由にする箇所のみ）
- 変更: `src/beme/scorer.py`（spec id の許可判定をレジストリ経由にする箇所のみ）
- 変更: `scripts/train_scorer.py`（`_extractor()` をレジストリ経由にする）
- 変更: `scripts/evaluate.py`（extractor 選択をレジストリ経由にする）
- 新規: `data/weights/ablations/*.json`（本タスクで作るのは下記 6 条件ぶんのみ）
- 新規: `scripts/build_ablation_weights.py`
- 新規: `tests/test_feature_registry.py`
- 必要な公開 export に限り: `src/beme/__init__.py`

`src/beme/lattice.py`・`converter.py`・`oracle.py`・`training.py`・`dictionary.py`・辞書ファイル・既存 `data/weights/legacy_v1.json`・`data/weights/decomposed_v1.json` は変更しない。

## 変更内容

### 1. レジストリ

`src/beme/features.py` に次を追加する。

```python
def resolve_spec(spec_id: str) -> FeatureSpec: ...
def extractor_for(spec_id: str) -> FeatureExtractor: ...
def registered_spec_ids() -> tuple[str, ...]: ...
```

- 既知 id は `legacy_v1`・`decomposed_v1` と、下記のアブレーション id。
- 未知 id は `ValueError`（`KeyError` でもよいが、既存の `unsupported feature specification: {id!r}` の文言と揃えること）。**黙って既定へフォールバックしてはならない。**
- レジストリは単一の定義表から作る。id と特徴集合の対応が 2 箇所に書かれる状態を作らないこと。`_SCRIPT_CATEGORIES` が宣言側と出力側の乖離を防いでいるのと同じ理由（`features.py:21-27` のコメント参照）。

### 2. アブレーション spec

基底 `decomposed_v1`（lexical 13・transition 16）から、次の 6 条件を派生させる。id は基底名 + `-` + 短い接尾辞。

| spec id | 基底からの差 |
|---|---|
| `decomposed_v1-lex_only` | transition 16 個をすべて除去 |
| `decomposed_v1-trans_only` | lexical 13 個をすべて除去 |
| `decomposed_v1-no_ipadic_cost` | `lex.ipadic_cost_mapped` を除去 |
| `decomposed_v1-no_source` | `lex.source.overlay`・`lex.overlay_cost` を除去 |
| `decomposed_v1-no_katakana` | `lex.katakana_surface` を除去 |
| `decomposed_v1-no_shape` | `lex.script.kanji` / `.hiragana` / `.katakana` / `.other` / `lex.surface_len` / `lex.proper_noun` を除去 |

`lex.base_cost` / `lex.unknown` / `lex.literal` は `-trans_only` を除きどの条件でも残す。`lex.unknown = 8.0` は未知語経路の暴走を抑える唯一の制約であり、`lex.base_cost` は raw cost を持たない overlay/v1 エントリの唯一のコスト源である。

### 3. 特徴フィルタ

`FeatureExtractor.lexical_features()` と `transition_features()` の返り値を、最後に `self.spec.lexical_features` / `self.spec.transition_features` の宣言集合で絞る。

- `legacy_v1` と `decomposed_v1` は全特徴を宣言しているので、**出力は 1 bit も変わらない**。
- `transition_features` は現状 `previous is None` のとき無条件に `{"trans.bos": 1.0}` を返す（`features.py:150-151`）。`-lex_only` では lexical が空 dict、`-trans_only` では transition が空 dict になること。空 dict は `_dot` で 0.0 になり、これが正しい挙動。
- `lexical_features` / `transition_features` の spec id 判定（`features.py:90` の `== "decomposed_v1"`、`:140` と `:147` の許可リスト）は、派生 spec でも `decomposed_v1` 系の抽出ロジックへ入るようにすること。id 文字列の等値比較を増やさず、`FeatureSpec` に「どの抽出系か」を示すフィールド（例 `family: str`）を持たせるのが安全。

### 4. 3 箇所の分岐をレジストリへ

- `src/beme/weights.py:106-111` — 未知 id に `FeatureSpec(feature_spec)` を作っている。これは lexical=`("lex.base_cost",)`・transition=空 の spec になるため、アブレーション重みファイルが「unknown feature name(s)」で全弾きされる。`resolve_spec()` に置き換える。
- `scripts/train_scorer.py:134-139` `_extractor()` — `extractor_for()` に置き換える。
- `scripts/evaluate.py:263-265` — **最重要**。現状は
  ```python
  extractor = (decomposed_v1_extractor() if weights.feature_spec == "decomposed_v1"
               else legacy_v1_extractor())
  ```
  で、未知 spec id を**黙って legacy_v1 へフォールバック**する。例外も警告も出ないまま、アブレーション条件の評価が別モデルの数値を出す。`extractor_for(weights.feature_spec)` に置き換え、未知 id は例外にすること。
- `src/beme/scorer.py` と `features.extractor_state_key`（`features.py:241-249`）の spec id 許可リストも派生 spec を受けるようにする。

### 5. アブレーション初期重みの生成

`scripts/build_ablation_weights.py` は `data/weights/decomposed_v1.json` を読み、各アブレーション spec について「宣言された特徴だけを残し、`feature_spec` を差し替えた」初期重み JSON を `data/weights/ablations/<spec_id>.json` へ書く。

- 出力は `Weights.dump` と同じ決定的 JSON（UTF-8・`ensure_ascii=False`・`sort_keys=True`・`indent=2`・末尾改行）。
- 除去された特徴の重み値は**捨てる**（0.0 として残さない）。残すと `Weights.load` の strict 検査で弾かれる。
- 同じコマンドの 2 回実行で byte-identical。
- `data/weights/legacy_v1.json` と `data/weights/decomposed_v1.json` は書き換えない。

## 必須テスト

- `legacy_v1` と `decomposed_v1` の `lexical_features` / `transition_features` の出力が、本タスク前と完全一致する（既存 golden で担保されるが、明示的な等価性テストも置く）。
- 各アブレーション spec で、宣言外の特徴名が返り値に 1 つも現れない。
- `-trans_only` の `lexical_features` が空 dict、`-lex_only` の `transition_features` が空 dict（`previous is None` の場合を含む）。
- `resolve_spec` / `extractor_for` が未知 id で例外。
- `Weights.load` が各アブレーション重みファイルを strict で読める。
- `Weights.load` が、アブレーション spec に対して除去済み特徴名を含む重みファイルを拒否する。
- `scripts/evaluate.py` にアブレーション重みを渡したとき、legacy_v1 へフォールバックせず、その spec の extractor が使われる（未知 id では非 0 終了）。
- `scripts/build_ablation_weights.py` の 2 回実行が byte-identical。
- `-trans_only` で `k_best_viterbi` を回すと 1 トークン経路が選ばれうる（PLAN §2.2(d) の縮退）。これは正しい挙動であり、例外にしない。

## 完了条件

- `legacy_v1` / `decomposed_v1` の全既存指標が 1 bit も変わらない。
- 6 つのアブレーション spec で学習と評価の両方が動く。
- spec id から extractor を引く経路が 1 本になり、未知 id が黙って別モデルへ落ちない。
- `uv run pytest -q` が全通し（現在 521 passed）。

## 検証コマンド

```bash
uv run pytest -q tests/test_feature_registry.py
uv run pytest -q tests/test_features.py tests/test_weights.py tests/test_scorer.py tests/test_scorer_golden.py tests/test_decomposed_scorer.py
uv run pytest -q

# 既存 4 指標が不変であること（Phase 2 baseline と一致するはず）
# 期待値: handmade NDCG@5=0.992972 Top-1=1.000000 / AJIMEE CER=0.290080 exact=0.070000 / Mozc 88/498
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json

# アブレーション重みの生成と読み込み
uv run python scripts/build_ablation_weights.py
uv run python scripts/evaluate.py --suite handmade --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/ablations/decomposed_v1-no_ipadic_cost.json
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- 対象外ファイルを変更しない。必要性が判明したら勝手に広げず報告する。
- `src/beme/training.py` の更新式・符号規約を触らない。アブレーションは extractor 側だけで表現する。
- 位置依存特徴を追加しない。`LinearScorer` の `edge.token` キーのキャッシュ前提（`scorer.py:40-52`）を壊さない。
- `data/dictionary*.tsv`・`data/d1/`・`data/weights/legacy_v1.json`・`data/weights/decomposed_v1.json` を生成・変更しない。
- 自動 formatter による対象外ファイルの一括変更をしない。
- テストを実装と同時に追加し、上記コマンドの実結果を報告する。
