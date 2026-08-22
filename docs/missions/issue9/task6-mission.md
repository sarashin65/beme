# Task 6: 重みの保存・読込と別プロセス再現（issue #9 Phase 1）

`docs/missions/issue9/PLAN.md` のタスク 6 を実装してください。§3.5（JSON 形式の設計）と §5 タスク6 を読んでから着手すること。

## 実装内容（PLAN.md タスク6 の要点）

1. `src/beme/weights.py`: §3.5 の JSON 形式の `load` / `dump` / 検証（`format` / `feature_spec` / 未知特徴 / `strict` フラグ / `weights_path` 予約フィールドの読み分岐スタブ）
2. `data/weights/legacy_v1.json` を生成してコミット。`legacy_v1` の重み定義の**唯一の出所**をこのファイルへ移す（コード内定数は読み込み失敗時のフォールバックにせず、ファイル欠損はエラーにする — 黙って別の目的関数で走らせない）
3. `scripts/evaluate.py` と `beme convert` に `--weights PATH`（既定 `data/weights/legacy_v1.json`）を追加
4. ラウンドトリップテスト: dump → load → 同一 dict、JSON バイト列が決定的（2 回 dump して同一）
5. **別プロセス再現テスト**: `subprocess.run([sys.executable, "-m", "beme", "convert", …, "--weights", tmp_path, "--json"])` を 2 通り（既定重み / 一時ファイルへコピーした同内容）で実行し、stdout が完全一致することを assert（受け入れ条件「学習済み重みを別プロセスで読み込み、同じ結果を再現できる」に直接対応）
6. 重み差し替えの健全性テスト: `trans.default` を 0.12 → 5.0 にした一時重みで 1-best が**変わる**ことを assert（重みが本当に効いていることの検証）

## 完了条件（PLAN.md より）

- [ ] `data/weights/legacy_v1.json` があり、それを使った `evaluate.py --suite all` が §1.5 の 4 指標と一致（CER=0.290080 / exact=0.070000 / Mozc 88/498 / handmade Top-1=1.000000）
- [ ] `feature_spec` 不一致・`format` 不一致・未知特徴名がそれぞれエラー
- [ ] 別プロセス実行の stdout が完全一致
- [ ] 重み改変で 1-best が変わる
- [ ] `uv run pytest -q` 全通（425 件 + 新規）

## 検証コマンド（PLAN.md より）

```bash
uv run pytest -q tests/test_weights.py -v
uv run python scripts/evaluate.py --suite all --reranker mock --weights data/weights/legacy_v1.json
uv run beme convert きしゃ --json --weights data/weights/legacy_v1.json
# 重み改変の健全性: trans.default を 5.0 にした一時ファイルで 1-best が変わること
```

## 制約

- git commit / push しない
- 作業はこの worktree（`~/work/beme-9`）内で行う
- 自己確認した出力（pytest 結果・evaluate 4指標・重み改変で 1-best が変わる確認・別プロセス一致確認）を貼って報告すること
