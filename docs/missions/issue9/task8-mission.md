# Task 8: `decomposed_v1` spec と Phase 1 レポート（issue #9 Phase 1・最終タスク）

`docs/missions/issue9/PLAN.md` のタスク 8 を実装してください。§3.4（`decomposed_v1` の設計）と §4.2-4（許容誤差版等価テスト）を読んでから着手すること。

## 実装内容（PLAN.md タスク8 の要点）

1. `decomposed_v1`（§3.4）を実装。初期重みは「legacy と同じ目的関数になる値」に設定し、それ以外は 0.0
2. 等価テスト（許容誤差版・§4.2-4）: golden 全ケースで
   - コスト差が `abs_tol=1e-9` 以内
   - **1-best surface が完全一致**、K=10 の surface 列が完全一致
3. `dictionary_v2.tsv` を使ったときの `evaluate.py --suite all` の 4 指標が §1.5 と一致することを確認（追加列があっても legacy 経路は変わらない）
4. `docs/missions/issue9/phase1-report.md` を作成:
   - Phase 1 チェックリスト 4 項目・受け入れ条件のうち Phase 1 該当分の充足状況
   - §1.5 と再測定値の対比表
   - 辞書 v2 のサイズ / ロード時間 / RSS 実測
   - `dictionary_learned_v2.tsv` baseline との関係（§2-C）
   - Phase 2 への引き継ぎ事項（`state_key` 設計・`Φ` 形式・`LatticePath` が位置を持たない件）

## 完了条件（PLAN.md より）

- [ ] `decomposed_v1` で 1-best と K=10 の surface 列が legacy と一致
- [ ] 両 spec で `Σ w·Φ == cost` 検算が通る
- [ ] レポートに Phase 1 チェックリスト 4 項目の証跡（コマンドと出力）が載っている
- [ ] `uv run pytest -q` 全通（435 件 + 新規）

## 検証コマンド（PLAN.md より）

```bash
uv run pytest -q
uv run python scripts/evaluate.py --suite all --reranker mock --weights data/weights/decomposed_v1.json
uv run python scripts/evaluate.py --suite all --reranker mock --dictionary data/dictionary_v2.tsv
```

## 制約

- git commit / push しない
- 作業はこの worktree（`~/work/beme-9`）内で行う
- 自己確認した出力（pytest 結果・decomposed_v1 等価確認・evaluate 4指標・レポートパス）を貼って報告すること
