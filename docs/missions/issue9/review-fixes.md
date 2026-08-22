# Task 8 修正: agy クロスレビュー指摘 3 件（issue #9 Phase 1）

PR #10 の agy クロスレビューで指摘された 3 件を修正してください。`docs/missions/issue9/PLAN.md` §3.4（decomposed_v1 の特徴量定義）と §3.5（重みファイル形式）を読んでから着手すること。

## 指摘 1（Critical）: CLI の feature_spec ハードコード

`src/beme/cli.py:207` で `LinearScorer(legacy_v1_extractor(), Weights.load(args.weights))` とハードコードされているため、`--weights data/weights/decomposed_v1.json` を指定すると `Weights.load` の既定期待値（legacy_v1）と不一致で `feature_spec mismatch` クラッシュする。

**修正**: `scripts/evaluate.py` と同様に、重み JSON から `feature_spec` を読み取って動的に extractor を切り替える（`Weights.load(args.weights, feature_spec=...)` に正しく渡す）。`beme convert --weights data/weights/decomposed_v1.json` がクラッシュせず動くことを確認するテストを追加。

## 指摘 2（Critical）: decomposed_v1 の宣言済み特徴量が未実装

`src/beme/features.py` の `lexical_features` で、`FeatureSpec.decomposed_v1` に宣言されている以下の特徴量が抽出されていない:

- `lex.source.overlay`: `token.source == "overlay"` の分岐で `{"lex.base_cost": token.cost}` を返すだけで、overlay 特徴が出ていない
- `lex.overlay_cost`
- `lex.reading_len` / `lex.surface_len` / `lex.script.*` / `lex.proper_noun`: ロジック未実装

特徴量ベクトルに現れないと Phase 2 の学習時に勾配が立たなくなる。

**修正**: PLAN §3.4 の定義に従い、decomposed_v1 で宣言された特徴量を正しく抽出する:
- `lex.source.overlay` = 1 if token.source == "overlay" else 0（ただし既存分岐との重複に注意。現在 raw_cost がある場合は `lex.ipadic_cost_mapped` と `lex.katakana_surface` を返して早期 return しているが、source == "overlay" の場合は overlay 特徴を出す）
- `lex.reading_len` = len(token.reading) / `lex.surface_len` = len(token.surface)
- `lex.script.*`: 表層の文字種（漢字・ひらがな・カタカナ・その他）に応じた 1-of-k（例: `lex.script.kanji` / `lex.script.hiragana` / `lex.script.katakana` / `lex.script.other`）
- `lex.proper_noun` = 1 if token.pos == "proper_noun" else 0
- 既に実装済みの `lex.unknown` / `lex.literal` / `lex.ipadic_cost_mapped` / `lex.katakana_surface` は維持

**重要**: 初期重み 0.0 の特徴は目的関数を変えないので、`decomposed_v1` の legacy 等価性（1-best surface 一致・cost abs_tol=1e-9）は保たれなければならない。既存の等価テストが引き続き通ること。

また、抽出された特徴量ベクトルに宣言済み特徴量が含まれることを確認するテストを追加（例: 辞書 v2 の overlay/ipadic エントリで `lexical_features` を呼び、`lex.source.overlay`・`lex.reading_len` 等が現れること）。

## 指摘 3（Warning）: 型アノテーションの抽象化漏れ

`src/beme/lattice.py:272` の `continuation: list[dict[str, float]]` は、状態キーが `scorer.state_key(edge)` 経由で `Hashable` に一般化されたため `list[dict[Hashable, float]]` が正しい。

**修正**: 型アノテーションを `Hashable` に変更（`Hashable` は既に import 済みか確認し、なければ追加）。

## 完了条件

- [ ] `uv run beme convert きしゃ --weights data/weights/decomposed_v1.json` がクラッシュせず動く
- [ ] decomposed_v1 の特徴抽出に宣言済み特徴量（source.overlay / reading_len / surface_len / script.* / proper_noun）が含まれるテストが pass
- [ ] 既存の decomposed_v1 等価テスト（1-best surface 一致・cost abs_tol=1e-9）が pass
- [ ] `uv run pytest -q` 全通（436 件 + 新規）
- [ ] `uv run python scripts/evaluate.py --suite all --reranker mock --weights data/weights/decomposed_v1.json` の 4 指標が変わらない（CER=0.290080 / exact=0.070000 / Mozc 88/498 / handmade Top-1=1.000000）

## 制約

- git commit / push しない
- 作業はこの worktree（`~/work/beme-9`）内で行う
- 自己確認した出力（pytest 結果・convert 実行結果・evaluate 4指標）を貼って報告すること
