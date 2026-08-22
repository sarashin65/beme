# Task 6: 評価データに対するリーク監査 CLI

## 目的

issue #9 Phase 3 注意点 5「Wikipedia 頻度と article-disjoint の出所を評価データ（AJIMEE / Mozc / handmade）に対して監査する」を実装する。

**d1 の manifest が既に主張している除外を追認するのではなく、独立実装で検証する。** 特に manifest が明記している「Mozc は読みと assertion のみで原文を持たないため、語彙共有は文書リークとみなさない」という判断は d1 生成側の主張であり、本タスクの追検証対象である。

最初に `docs/missions/issue9-phase3/PLAN.md` の §2.3・§3.5、`data/d1/manifest.json`、`data/d1/data_gate.json`、`src/beme/training_data.py`、`src/beme/benchmark.py`（`load_ajimee_cases` / `load_mozc_cases`）、`src/beme/evaluation.py`（`load_cases`）を読むこと。

## 依存

なし。Task 1〜5 と並行して進めてよい。

## 前提（実測済み・信じてよい）

`data/d1/manifest.json` と `data/d1/data_gate.json` は **repo に同梱されている**（`.gitignore:8` が除外するのは `data/d1/*.jsonl` だけ）。したがって manifest の検証は miniPC で完結する。`*.jsonl` 本体を要する検証だけが GPU マシン（Task 8）行きになる。

manifest が主張している内容:

| 項目 | 主張値 |
|---|---|
| 記事 disjoint | `train_dev` / `train_test` / `dev_test` すべて重複 0 |
| 正規化グループ fingerprint 重複 | 3 ペアとも 0 |
| ユニーク記事 | train 696 / dev 65 / test 129 |
| ユニーク読み | train 8,964 / dev 976 / test 1,925 |
| 分割アルゴリズム | `sha256(seed:article_id) % 10000`・seed 20260822・train<7700≤dev<8500≤test |
| 評価重複除外 | AJIMEE original/context と handmade context の正規化部分文字列・最小 12 文字・238 term・記事 3 本を除外 |
| Mozc | 「読みと assertion のみで原文を持たないため語彙共有は文書リークとみなさない」 |
| ラベル監査 | 200 件抽出・正解 199・`label_accuracy=0.995` |
| 出所 | `wikimedia/wikipedia` config `20231101.ja` revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |

## 対象ファイル

- 新規: `scripts/audit_leakage.py`
- 新規: `tests/test_audit_leakage.py`

`src/beme/` 配下・`data/` 配下は変更しない。

## 変更内容

### 1. manifest 内部整合の独立検証（miniPC で完結）

manifest / data_gate の主張が自己矛盾していないかを、manifest を信じずに再計算する。

- `split.algorithm` を独立実装し、`split.ranges` と `split.seed` から各 article_id の split を再計算する。`*.jsonl` があれば実データの `split` フィールドと照合し、無ければ「照合不能」と明示する（推測で埋めない）。
- `counts` / `unique_articles` / `unique_readings` / `contrast_pairs` の相互整合（例: ユニーク記事数 ≤ 件数、ユニーク読み数 ≤ 件数）。
- `data_gate.gates` の各 bool が、同じ JSON 内の対応する数値と矛盾しないこと（例: `article_disjoint=true` なのに `article_overlaps` が空でない、といった不整合の検出）。
- `audit.label_accuracy == correct / audited` の再計算。

### 2. 評価データに対する重複監査（miniPC で完結する部分）

`data/ajimee_bench_jwtd_v2_v1.json`（200 件）、`data/mozc_oss.tsv`（498 件）、`data/evaluation_cases.jsonl`（14 件）を読み、次を出す。

- **表層 n-gram 重複**: 各評価ケースの参照表記・context から正規化（NFKC + 空白除去）した文字 n-gram（n = 12・manifest の `minimum_normalized_characters` に合わせる）を作り、d1 の `*.jsonl` がある場合は d1 側の gold 表記・context と照合する。
- **読みの重複**: 評価ケースの読みと d1 の読みの完全一致数。これは**リークではなく分布の重なり**なので、必ず別項目として出し、ラベルもそう付ける（`reading_overlap`、`document_leakage` とは別フィールド）。
- **語彙の重複**: 評価ケースの gold 表記に現れる語が d1 train の gold 表記に現れる割合。これも**リークではない**（辞書ベースの IME で語彙が重なるのは当然）。同様に別フィールド。

**この 3 つを区別せずに 1 つの「リーク率」にまとめてはならない。** manifest が Mozc について述べている判断（語彙共有はリークでない）が妥当かどうかは、この分離ができて初めて議論できる。

### 3. Mozc の判断の追検証

manifest の主張「Mozc は読みと assertion のみで原文を持たない」を、`data/mozc_oss.tsv` の実データで確認する。

- 各ケースが持つフィールド（`id` / `label` / `reading` / `expected` / `command` / `expected_rank`）を数え、原文・文脈に相当する列が実在しないことを示す。
- 主張が正しければ「文書リークの経路が構造上存在しない」ことの根拠になる。正しくなければ**主張の反証として報告する**（manifest を書き換えない）。

### 4. 出力

```json
{
  "manifest_consistency": {"checks": [{"name": "...", "passed": true, "detail": "..."}], "all_passed": true},
  "split_recomputation": {"verified": true | false, "reason": "..."},
  "evaluation_overlap": {
    "ajimee": {"document_leakage": {...}, "reading_overlap": {...}, "vocabulary_overlap": {...}},
    "mozc":   {...},
    "handmade": {...}
  },
  "mozc_source_document_claim": {"claim": "...", "verified": true | false, "fields_present": [...]},
  "d1_jsonl_available": true | false,
  "unverifiable": ["..."]
}
```

`unverifiable` に、`*.jsonl` が無いために検証できなかった項目を**必ず列挙する**。空配列を装わないこと。

決定性: UTF-8・`ensure_ascii=False`・`sort_keys=True`・`indent=2`・末尾改行、atomic replace、2 回実行で byte-identical。

## 必須テスト

- 合成 manifest で、各整合チェックが意図どおり pass / fail する（意図的に矛盾させた fixture を用意する）。
- `split.algorithm` の再実装が、manifest の `seed` と `ranges` から既知の article_id に対して期待する split を返す。
- 文書リーク・読み重複・語彙重複が別フィールドで出る。
- `*.jsonl` が無い環境で `d1_jsonl_available: false` になり、`unverifiable` に該当項目が入り、**非 0 終了しない**（miniPC で部分監査として動くこと）。
- Mozc の主張検証が、原文列を持つ合成 TSV に対して `verified: false` を返す。
- 2 回実行が byte-identical。

## 完了条件

- miniPC で（`*.jsonl` 無しで）manifest 整合と評価データ側の監査が完走する。
- `*.jsonl` を要する項目が `unverifiable` に明示される。
- 文書リーク / 読み重複 / 語彙重複が混同されない形で出力される。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_audit_leakage.py
uv run python scripts/audit_leakage.py --help
uv run pytest -q

# miniPC（data/d1/*.jsonl が無い状態での部分監査）
uv run python scripts/audit_leakage.py \
  --d1-dir data/d1 \
  --ajimee-cases data/ajimee_bench_jwtd_v2_v1.json \
  --mozc-cases data/mozc_oss.tsv \
  --cases data/evaluation_cases.jsonl \
  --output /tmp/leak_audit.json --json

# GPU マシン（*.jsonl 込みの完全監査）は Task 8 で実行する
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- `src/beme/` 配下・`data/` 配下を変更しない。**manifest を書き換えない。** 主張が誤っていれば報告する。
- `data/d1/*.jsonl` を生成しない・推測で合成しない。
- 検証できなかった項目を「問題なし」と書かない。`unverifiable` に列挙する。
- 文書リーク・読み重複・語彙重複を 1 つの指標に混ぜない。
- ネットワーク・GPU・外部モデルを必要としない CLI/テストにする。
- テストを実装と同時に追加し、上記コマンドの実結果を報告する。
