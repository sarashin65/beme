# Mission 5: A* 性能問題の修正（厳密後向き DP + ヒープキー最適化）

## 指示

Opus の調査（probe3/6 で実証済み）に基づき、以下の修正を実装してください。対象: `~/work/beme-p2`（branch: feat/p2-kbest-and-exact）

## 背景（調査結果の要約）

- **P0（必須）**: `backward_viterbi` の下界が「許容的だが緩い」。`continuation[pos][prev_pos]` が「端末に直行する短いエッジ」で長さ情報を潰し、残余コストを過小評価（実データで約 0.15/文字の緩み）→ A* が全展開して爆発
  - 修正: 厳密な後向き DP `R[pos][prev_pos]`（POS 17 種なので O(nodes × 17 × edges)・無視できるコスト）を新関数として追加
  - **互換性注意**: 既存の `backward_costs` / `backward_viterbi`（`tuple[float,...]` を返す公開 API）はテストがあるので**残す**。A* 用に新関数（例: `backward_costs_by_pos`）を追加
  - 実測: 厳密 R[pos][prev_pos] に差し替えると len45: 25s 未完了 → 0.023s / pops 353
- **P1（必須）**: ヒープキーから `surface` / `tokens_key` を削除 → `(f, cost, seq, end, child)` に。P0 後も 3〜7 倍の差（決定性は壊れない: `build_lattice` がエッジをソート済み + converter.py で最終再ソート）
- **P2（推奨）**: `_tokens_cache` を削除（P1 後は探索中に `tokens` が呼ばれないため不要・メモリ 2.1GB の増幅要因）

## 完了条件

- 既存テスト全件パス（`uv run pytest -q`）
- 性能実測: `scripts/evaluate.py --suite ajimee --json` が**数十秒以内**で完了（修正前は 300s+ 未完了）
- `scripts/evaluate.py --suite mozc --json` が OOM せず完了（修正前は 3.8GB で kill）
- AJIMEE の mean CER が **0.2901 前後を維持**（品質無傷の確認）
- 評価結果（タイム・CER・pass 数）を報告に含める

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- 実装後は自己確認（テスト・評価実行）して報告
