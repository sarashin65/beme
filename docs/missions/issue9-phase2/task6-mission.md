# Task 6: 実データ最終検証・README・Phase 2 レポート

## 目的

供給済みの固定 train/dev/test で再現可能な最終学習を 1 回行い、held-out test baseline 比較、coverage/model error、handmade のケース別回帰を記録して Phase 2 を引き渡す。

最初に `README.md`、`docs/missions/issue9-phase2/PLAN.md` 全体、`docs/missions/issue9/phase1-report.md`、Task 1–5 の成果を読むこと。

## 依存・開始条件

- Task 1–5 が完了し `uv run pytest -q` が通ること。
- `data/d1/train.jsonl`（想定 10k）、`dev.jsonl`（1k）、`test.jsonl`（2k）が供給されていること。
- 3 ファイルが無い、schema が異なる、provenance が不明な場合は**生成して代用せず blocked と報告**する。

## 対象ファイル

- `README.md`: Phase 2 学習・再現・評価コマンド、データ前提を追記
- 新規: `docs/missions/issue9-phase2/phase2-report.md`
- 必要なら `.gitignore`: 学習 artifact directory だけを限定追加

製品コード、tests、データ JSONL、辞書、既存 weights は変更しない。学習出力は `artifacts/issue9-phase2/` に置き、明示指示なしにモデル weights を git 管理対象へ追加しない。

## 実行内容

1. loader で3 split を検証し、件数、SHA-256、source、article disjoint 判定、正解 key（`target_surface`/`positive`）を記録する。
2. PLAN の固定例 `epochs=5, learning_rate=0.1, seed=9` で学習する。これらを test を見て変更しない。
3. 同じコマンドを独立出力先へ2回実行し weights/report の `cmp` を取る。
4. report から baseline/learned の dev/test coverage、model、end-to-end error を対比する。
5. learned weights を別 process の `scripts/evaluate.py --suite all --reranker mock` へ渡す。
6. handmade report の `per_query` を baseline と learned で比較し、14件すべてを表にする。aggregate が同じでも順位/候補変化を隠さない。
7. AJIMEE/Mozc は Phase 2 の補助回帰として記録し、test tuning に使わない。
8. full pytest を最終実行する。

## phase2-report.md 必須項目

- commit、辞書/初期 weights/split SHA-256、全 CLI 引数。
- split 件数、provenance、article disjoint、MeCab かな化と Wikipedia 分布の注意。
- epoch ごとの train/dev 指標、dev 選択規則と選択 epoch。
- held-out test の baseline vs learned（coverage/model/end-to-end）。
- 学習済み weights の path/hash/metadata と別 process 再現結果。
- handmade 14件の case-by-case before/after、AJIMEE、Mozc。
- 失敗/coverage 例の代表サンプル（個人情報や巨大列挙なし）。
- Phase 3 引き継ぎ（ablation、cost scale 再校正、oracle recall@k、data bias、配布）。

## 完了条件

- 同一コマンド2回の weights/report が byte-identical。
- test は dev 選択後にのみ評価され、baseline 比較が報告される。
- coverage/model error が分離される。
- learned weights が別 process で読み込まれ、同じ結果を返す。
- handmade 回帰をケース単位で説明する。
- full pytest と既存 suite が完走する。

## 検証コマンド

```bash
uv run pytest -q
uv run python scripts/train_scorer.py \
  --train data/d1/train.jsonl --dev data/d1/dev.jsonl --test data/d1/test.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --initial-weights data/weights/decomposed_v1.json \
  --epochs 5 --learning-rate 0.1 --seed 9 \
  --output-weights artifacts/issue9-phase2/run1/weights.json \
  --output-report artifacts/issue9-phase2/run1/report.json
uv run python scripts/train_scorer.py \
  --train data/d1/train.jsonl --dev data/d1/dev.jsonl --test data/d1/test.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --initial-weights data/weights/decomposed_v1.json \
  --epochs 5 --learning-rate 0.1 --seed 9 \
  --output-weights artifacts/issue9-phase2/run2/weights.json \
  --output-report artifacts/issue9-phase2/run2/report.json
cmp artifacts/issue9-phase2/run1/weights.json artifacts/issue9-phase2/run2/weights.json
cmp artifacts/issue9-phase2/run1/report.json artifacts/issue9-phase2/run2/report.json
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights artifacts/issue9-phase2/run1/weights.json --json
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。Hermes が検証後に commit する。
- 対象外ファイルを変更しない。特に train/dev/test、辞書、既存 weights を書換えない。
- test/handmade/AJIMEE/Mozc を見て epoch、rate、seed、特徴を選び直さない。
- データが無ければ synthetic や別 corpus を「最終結果」として代用しない。
- 実測値を省略・丸め替え・期待値で補完しない。コマンドと出力を report に残す。
