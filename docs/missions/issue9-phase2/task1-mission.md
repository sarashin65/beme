# Task 1: 正解表記制約付きオラクル DP

## 目的

`(reading, target_surface)` について、正解表記を生成できる全ラティス経路のうち、現在の `Scorer` で cost 最小の latent gold path を返す。K-best の幅で代用せず、coverage 不足を `None` として明示する。

最初に `README.md`、`docs/missions/issue9-phase2/PLAN.md` §1–2.1、`docs/missions/issue9/PLAN.md` §3.7、`docs/missions/issue9/phase1-report.md` を読むこと。

## 対象ファイル

- 新規: `src/beme/oracle.py`
- 新規: `tests/test_oracle.py`
- 必要な公開 export に限り: `src/beme/__init__.py`

`src/beme/lattice.py`、`converter.py`、scorer/features/weights、辞書、scripts は変更しない。

## 変更内容

1. `constrained_best_path(lattice, target_surface, scorer) -> LatticePath | None` を実装する。
2. DP state は `(input_position, target_position, previous_state)`。previous state は必ず `scorer.state_key(edge)` を使う。
3. edge.surface が target の現在位置に exact prefix match するときだけ遷移する。
4. edge/transition cost は既存探索と同じ加算順にする。
5. 同じ state の複数経路は current model cost 最小を残す。tie は既存 `_path_key` と同じ `(cost, surface, token reading/surface/pos/cost 列)` で決定的にする。
6. 完成 path は既存 `LatticePath`。該当経路がなければ `None`。正解境界は入力に要求しない。
7. private な K-best の有限幅フィルタで実装しない。Unicode 正規化も暗黙に行わない。

## 必須テスト

- 一意な gold path。
- 同じ target surface を異なる分割で生成でき、scorer weight を変えると latent path が変わる。
- target が生成不能なら `None`。
- reading 全消費/target 全消費の片方だけでは成功しない。
- POS 以外の custom Hashable state key を使う fake scorer。
- 同 cost tie が辞書挿入順に依存しない。
- negative edge/transition cost でも正しい gold path。
- 返した path の `surface`, `reading`, `cost` が scorer に一致する。

## 完了条件

- オラクルの結果が小さな exhaustive enumeration の最良 exact-surface path と一致する。
- coverage miss が例外や空 path ではなく `None` になる。
- 既存 K-best/Converter API と golden を変更しない。

## 検証コマンド

```bash
uv run pytest -q tests/test_oracle.py
uv run pytest -q tests/test_dictionary_lattice.py tests/test_scorer_golden.py
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- 対象外ファイルを変更しない。必要性が判明したら勝手に広げず報告する。
- 既存探索コアの K-best/後方 DP と `Converter` の外部 interface を後方互換のまま保つ。
- 位置依存特徴を追加しない。`LinearScorer` の token-key cache 前提を壊さない。
- テストを実装と同時に追加し、上記コマンドの実結果を報告する。
- 自動 formatter による対象外ファイルの一括変更をしない。

