# Task 5: 層別評価とケース単位 before/after 診断 CLI

## 目的

issue #9 Phase 3 注意点 4（層別評価）と注意点 7（ケース単位の before/after 診断）を実装する。

**注意点 7 は Phase 2 の失敗への直接の対応である。** Phase 2 は handmade の全ケースの順位変動を記録しなかったため、`hashi-no`（`はしの` が `橋の` → `橋野`）の回帰説明が「Wikipedia 由来の頻度・固有名詞バイアスと `の` の境界選好の変化と推測」という仮説どまりになった。本タスクの成果物は、この種の回帰を**特徴ごとの寄与分解として数値で決着させる**ためのものである。

最初に `docs/missions/issue9-phase3/PLAN.md` の §3.5・§3.7、`src/beme/features.py`（特に `path_features` の weights 引数）、`src/beme/converter.py`、`src/beme/evaluation.py`、`src/beme/benchmark.py`、`src/beme/training_data.py` を読むこと。

## 依存

なし。Task 1〜4 と並行して進めてよい。

## 対象ファイル

- 新規: `scripts/diagnose_cases.py`
- 新規: `tests/test_diagnose_cases.py`

`src/beme/` 配下は変更しない。特に `benchmark.py` の `per_case` 出力を変えないこと（既存評価の出力を変えると Phase 2 との比較が壊れる）。必要な値は `Converter.convert` を直接呼んで取る。

## 変更内容

### 1. ケース単位 before/after 診断

`--baseline-weights` と `--learned-weights` の 2 条件で同じデータを変換し、次を出す。

1. **全ケースの順位表**（Top-1 が変わらなかったケースも省略しない）
   - ケース id、読み、gold 表記
   - baseline / learned それぞれの Top-1 surface、gold の順位（圏外は `null`）、Top-5 の `(surface, cost)`
   - 判定: `unchanged` / `fixed`（誤→正）/ `regressed`（正→誤）/ `still_wrong`
2. **特徴レベルの寄与分解**（Top-1 が変わったケースのみ）
   - `path_features(baseline_top1, extractor, baseline_weights)` と `path_features(learned_top1, extractor, learned_weights)`
   - 各特徴について `w[f] * Δf` を計算し、**絶対値の降順**で並べる
   - 「どの特徴が何点分ひっくり返したか」が読める形にする

**`path_features` には必ずスコアリングに使った `Weights` を渡すこと**（`features.py:180-194`・Phase 1 コメント項目 2）。baseline と learned で別の重みを使うので、渡し間違えると `ValueError` になる（黙って壊れはしないが、渡し忘れると落ちる）。

寄与分解は「baseline の重みで両経路を評価した差」と「learned の重みで両経路を評価した差」の**両方**を出すこと。前者は「経路が変わったこと」の効果、後者は「重みが変わったこと」の効果で、混ぜると解釈できない。

### 2. 層別評価

d1 の split ファイル（`--data`）に対して、次の 4 軸で error rate を層別する。層はレコードから機械的に決まるものだけを使い、目視分類を入れない。

| 軸 | 層 |
|---|---|
| 読み長 | `len(reading)` を `1-3` / `4-6` / `7-10` / `11-16` |
| 固有名詞 | gold path のトークンに `pos == "proper_noun"` を含むか（2 層） |
| 数値読み | 読みが数詞（`いち に さん し ご ろく なな はち きゅう じゅう ひゃく せん まん`）を含むか、または gold 表記が `[0-90-9]` を含むか（2 層） |
| span_tokens | レコードの `span_tokens`（1 / 2 / 3）別、および `contrast_pair_id` の有無別 |

**`span_tokens` と contrast pair は「MeCab 変換品質」の代理指標であって、変換品質そのものの層別ではない。** 真の品質は d1 の 200 件抽出監査（`label_accuracy=0.995`）しか根拠が無い。出力 JSON にこの但し書きを機械可読なフィールド（例 `"proxy_metric": true` と `"caveat": "..."`）として入れ、レポートが取り違えられないようにすること。

各層について baseline / learned の `end_to_end_error_rate` と件数を出す。**件数が 30 未満の層は率を出しつつ `"low_support": true` を立てる**（1 例の差が率を大きく動かすため）。

### 3. 代表誤りの抽出

d1 test について、learned で新たに誤りになった例（`regressed`）と新たに正解になった例（`fixed`）を、**レコード id の昇順で各 20 件**。無作為抽出しない（seed 依存の結果をレポートに載せないため）。

### 4. CLI

```text
--data                 gold を持つ JSONL（d1 split）。--data か --suite のどちらか必須
--suite                {handmade, mozc}
--dictionary           既定 data/dictionary_v2.tsv
--baseline-weights     必須
--learned-weights      必須
--top-k                既定 5
--beam-width           既定 64
--stratify             層別を出す（--data 指定時のみ有効）
--output               JSON 出力先（必須）
```

- AJIMEE は文単位変換のため単一読みの順位表と定義が合わない。`--suite ajimee` は明示的なエラーで拒否する。
- 出力は UTF-8・`ensure_ascii=False`・`sort_keys=True`・`indent=2`・末尾改行、一時ファイル経由の atomic replace。
- wall clock・絶対パス・ランダム temp 名を出力に入れない。同じ入力の 2 回実行が byte-identical。

## 必須テスト

- 合成ケースで、Top-1 が変わらないケースも順位表に出る。
- 下位候補だけが動いたケースが `unchanged` として記録され、かつ Top-5 の差分が残る。
- `fixed` / `regressed` / `still_wrong` / `unchanged` の 4 判定が正しく振られる。
- 寄与分解の総和が、2 経路のコスト差と一致する（浮動小数の許容誤差内）。**これが分解の正しさの検算になる。**
- 「経路が変わった効果」と「重みが変わった効果」が別フィールドで出る。
- `path_features` に誤った weights を渡すと `ValueError` になることを確認するテスト（渡し忘れの回帰防止）。
- 層別の各層の件数の総和が全件数と一致する（層の網羅性と排他性）。
- 件数 30 未満の層に `low_support` が立つ。
- 代表誤りが id 昇順で決定的。
- `--suite ajimee` が非 0 終了。
- 2 回実行が byte-identical。

## 完了条件

- handmade 14 件の全ケース順位表が出る（Phase 2 が出せなかったもの）。
- Top-1 が変わったケースについて、特徴ごとの寄与が数値で出る。
- d1 split に対して 4 軸の層別が出る。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_diagnose_cases.py
uv run python scripts/diagnose_cases.py --help
uv run pytest -q

# handmade の before/after（repo 内データのみ。learned 重みが無ければ baseline を両方に渡して
# 「差分ゼロ」が正しく出ることを確認する）
uv run python scripts/diagnose_cases.py --suite handmade \
  --dictionary data/dictionary_v2.tsv \
  --baseline-weights data/weights/decomposed_v1.json \
  --learned-weights data/weights/decomposed_v1.json \
  --output /tmp/diag_handmade.json

# 層別は d1 が要るので GPU マシン（Task 8）で実行する
# uv run python scripts/diagnose_cases.py --data data/d1/test.jsonl --stratify \
#   --dictionary data/dictionary_v2.tsv \
#   --baseline-weights data/weights/decomposed_v1.json \
#   --learned-weights artifacts/issue9-phase3/A0/seed9/weights.json \
#   --output artifacts/issue9-phase3/diag/A0-test.json
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- `src/beme/` 配下を変更しない。特に `benchmark.py` / `evaluation.py` の出力形式を変えない。
- `data/d1/*.jsonl` を生成しない・推測で合成しない。テストは repo 内の小さな合成 fixture で完結させる。
- 無作為抽出を使わない。すべて決定的な順序で出す。
- 代理指標を真の指標として出力しない（`span_tokens` の但し書きを必ず入れる）。
- ネットワーク・GPU・外部モデルを必要としない CLI/テストにする。
- テストを実装と同時に追加し、上記コマンドの実結果を報告する。
