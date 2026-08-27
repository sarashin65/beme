# task1 レポート書き込み + レビュー（Opus）

あなたは BeME issue #19 の task1（S 実測）のレポート執筆とレビューを担当します。worktree `/home/sarashin/work/beme-19`（branch `feat/issue19-f2`）で作業してください。

## 背景

task1 の実装・測定は完了済みです。残っているのは:
1. **レポート未執筆**: `docs/missions/issue19/s-measurement-report.md` がまだ無い
2. **レビュー**: 実装（`scripts/measure_local_ceiling.py`・`tests/test_measure_local_ceiling.py`）と測定結果の品質確認

## 完了済みの成果物（触らない・検証のみ）

- `scripts/measure_local_ceiling.py`（661行・Ornith 実装・syntax OK）
- `tests/test_measure_local_ceiling.py`（5件・全 pass）
- pytest: 643 passed / 2 failed（失敗は `tests/test_audit_leakage.py` の環境固有 2 件・`data/d1/*.jsonl` がこの worktree にあるため・直さない）
- `artifacts/issue19/s-measurement/ties.json`（SHA-256: f5972c811ba86ac977d81fa1d02aa1daa72de148858a3cdff5847eb317bf9ef8・12,000 ケース完走）
- `artifacts/issue19/s-measurement/coverage_pairs.jsonl`（SHA-256: 6dc370d5e87c77326838f4427cbf657cd4be4b2c860c29401ed4a0f5857a00e2）
- `artifacts/issue19/s-measurement/smoke.json`（SHA-256: c063ed05d4dc0880e401e24ecc005ab28a4b6d4556a61c62dd1937bc5805283d）
- `data/d1/{train,test}.jsonl`（gitignore・ローカルにあり）

## やること

### 1. レビュー（先に実施）

`scripts/measure_local_ceiling.py` と `tests/test_measure_local_ceiling.py` を読み、以下を確認してください:
- 4 粒度（lex14/lex9/full25/cost）× 2 候補集合（lattice/jsonl）× 3 率の測定が PLAN §3.1 の定義どおりか
- bit-exact キー（`repr()` ベース）とコスト同点の `math.isclose(rel_tol=0, abs_tol=1e-12)` の使い分けが妥当か
- gold パスの取り方（`constrained_best_path`）・`gold_unreachable` の分母除外が正しいか
- テストが意味のある検証をしているか（形だけのテストでないか）
- `src/beme/` が 1 行も変更されていないこと（`git diff --stat origin/main -- src/beme/` が空）

問題があれば修正してください（`src/beme/` は変更禁止・`scripts/` と `tests/` は修正可）。

### 2. レポート執筆

`docs/missions/issue19/s-measurement-report.md` を書いてください。**ties.json の中身を読み、実測値から正確に**書くこと（推測禁止）。含める内容:

1. 先頭に結論と数値（PLAN §3.5 の分岐 A・B のどれに該当したか・根拠の数字を明示）
2. 4 粒度 × 2 候補集合 × 3 率の表（train / test）
3. 被覆率の表（S2）
4. **Opus アドバイザリーの辞書レベル 24.9% と d1 候補集合上の lex14 率の比較と説明**（辞書は多候補読み 72,013・d1 は読み 9,778 種・異なる分布から引かれている等）
5. 分岐 A / B の判定と、それが F2 の期待値に何を意味するか
6. 実行環境（機名 WIN-I7DHB6T7L79・Python 3.11.15・git commit bd12789）・入力の SHA-256
7. 単一 split の値に p 値ラベルを付けない（Phase 3/4 のレポート規約）

PLAN §3.5 の分岐基準:
- 分岐 A: gold_vs_nongold_tied_rate（粒度 c・v2 lattice・test）: ≥10% = 想定どおり / 3-10% = 進めるが理論上限を事前登録 / <3% = 本命扱いを外す
- 分岐 B: gold 到達不能率: 5% 超なら辞書側が先（ただし F2 実装は止めない）
- 分岐 C: (読み,表記) コーパス被覆率: ≥70% = そのまま / 50-70% = 走査量4倍+再ビルド / <50% = 走査量4倍+4特徴目

### 3. 検証

- `uv run pytest -q` で 643 passed / 2 failed を確認
- `git diff --stat origin/main -- src/beme/` が空
- `git diff --check`

## 制約

- `src/beme/lattice.py` / `converter.py` 変更禁止（探索コア）
- `git commit` / `git push` しない
- この worktree の中だけで作業
- 日本語で報告。数値と結論を先頭に

完了したら、レポートのパスと結論（分岐判定・主要数値）を報告してください。
