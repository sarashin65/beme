# Task 7: 経路特徴ベクトル `Φ(P)` の抽出・保存（issue #9 Phase 1）

`docs/missions/issue9/PLAN.md` のタスク 7 を実装してください。§3.6（Φ(P) の定義）と §5 タスク7 を読んでから着手すること。

## 実装内容（PLAN.md タスク7 の要点）

1. `features.path_features(path, extractor)` を実装（§3.6）
2. `scripts/dump_features.py`: 読みリストを受けて K-best 各経路の `Φ` を JSON Lines へ出力
3. **一致検算テスト**: 任意のラティス・任意の K に対し `Σ w[f]·Φ[f] == path.cost`（legacy は `==`、その他 spec は `abs_tol=1e-9`）
4. 差分ユーティリティ `feature_delta(gold, pred) -> dict[str, float]`（Phase 2 の更新式が直接使う）。0 になる項を落とすこと・キー順が決定的であることをテスト
5. JSON Lines のラウンドトリップ（保存 → 読込 → 同一 dict）

## 完了条件（PLAN.md より）

- [ ] golden の全ケース・全順位で `Σ w·Φ == cost`
- [ ] `feature_delta` が対称性（`delta(a,b) == -delta(b,a)`）と決定的キー順を満たす
- [ ] `dump_features.py` の出力が 2 回実行でバイト単位一致
- [ ] `uv run pytest -q` 全通（432 件 + 新規）

## 検証コマンド（PLAN.md より）

```bash
uv run pytest -q tests/test_features.py -v
uv run python scripts/dump_features.py --cases tests/data/golden_readings.txt --k 5 --output /tmp/phi1.jsonl
uv run python scripts/dump_features.py --cases tests/data/golden_readings.txt --k 5 --output /tmp/phi2.jsonl
cmp /tmp/phi1.jsonl /tmp/phi2.jsonl
```

## 制約

- git commit / push しない
- 作業はこの worktree（`~/work/beme-9`）内で行う
- 自己確認した出力（pytest 結果・Σw·Φ==cost の確認・dump 2 回の cmp 結果）を貼って報告すること
