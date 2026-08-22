# Task 5: Phase 2 横断統合・後方互換テスト

## 目的

オラクル、学習、永続化、既存 exact search を一連で検証し、個別 unit test では見逃す cache、負 cost、別プロセス、既存回帰を gate にする。

最初に `README.md`、`docs/missions/issue9-phase2/PLAN.md` §2.5・§4–5、issue #9 の Phase 1 owner コメント、Task 1–4 の実装を読むこと。

## 依存

Task 1–4 が完了していること。

## 対象ファイル

- 新規: `tests/test_phase2_integration.py`
- 必要な小規模 fixture: `tests/data/phase2/`
- バグ修正が必要なら Task 1–4 の新規ファイルだけ。既存コア変更は先に報告する。

## 変更内容・必須シナリオ

1. synthetic train/dev/test を CLI で学習し、保存 weights を別 Python process で読み込む。
2. 同じ reading の baseline と learned 1-best を比較し、学習 signal が実際に探索へ届くことを確認する。
3. learned weight に負値がある状態で `backward_costs_by_state` + `k_best_viterbi` が exhaustive enumeration と同じ順序になる小ラティスを検証する。
4. constrained oracle の latent path、prediction、`path_features(..., current_weights)` の cost 再構成を一つのテストで検証する。
5. coverage-only、model-error-only、correct の3例を混ぜ、report 集計を検証する。
6. legacy default scorer と `Converter` の引数なし呼び出しが従来通り動くことを確認する。
7. `scripts/evaluate.py --weights <learned>` が読み込める smoke test を行う。実モデル reranker は使わない。
8. 既存 golden K=10、dictionary v1/v2、legacy/decomposed の回帰テストを再実行する。

## 完了条件

- Phase 2 の全 acceptance contract が synthetic end-to-end で通る。
- negative weights を辞書 cost の負値で実現しない。
- `lattice.py` の探索ロジックと `Converter` public interface に変更がない。
- full pytest が全通する。

## 検証コマンド

```bash
uv run pytest -q tests/test_phase2_integration.py
uv run pytest -q tests/test_scorer_golden.py tests/test_decomposed_scorer.py tests/test_converter.py
uv run pytest -q
uv run python scripts/evaluate.py --suite all --reranker mock
```

既存 baseline 期待値:

- Handmade: NDCG@5 0.992972 / MRR@5 1.000000 / Top-1 1.000000
- AJIMEE: CER 0.290080 / exact 0.070000
- Mozc: 88/498

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- 原則テスト/fixture 以外を変更しない。製品コードの欠陥を見つけた場合は最小修正と理由を報告する。
- golden や baseline 値を新出力へ合わせて更新して回帰を隠さない。
- 外部ネットワーク、GPU、LLM/Cross-Encoder 実モデルを使わない。
- 全検証コマンドの実結果を報告する。

