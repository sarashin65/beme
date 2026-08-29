# task3 v2 修正（Opus レビュー指摘対応）

BeME issue #19 task3（F2-a コーパス頻度表ビルダー）の v2 実装を、Opus レビューの指摘に従って修正してください。作業ディレクトリ: `/home/sarashin/work/beme-19`（branch `task3-v2`）。

## 現状

- v2 実装（luna 実装・Opus 計画準拠）は Opus レビューで「条件付き合格」
- pytest 654 passed / 2 failed（失敗は既知の audit_leakage のみ・直さない）
- `data/corpus_freq_v1.bin`（4.7MB）は生成済みだが **manifest は v1 の遺物のまま**（bin と不一致）
- リーク対策 3 層は健全（触らない）

## 修正項目（Opus レビュー指摘）

### M-A（必須・最優先）: `scripts/build_corpus_freq.py:179` の `marginals_consistent` が O(S×P) で 10.7 分
現在:
```python
all(sum(count for (r, s), count in pair_counts.items() if s == surface) == count
    for surface, count in surface_counts.items())
```
→ O(P) に修正:
```python
from_pairs = Counter()
for (reading, surface), count in pair_counts.items():
    from_pairs[surface] += count
marginals_consistent = (from_pairs == surface_counts
                        and sum(pair_counts.values()) == total_tokens)
```
（`Counter` は既に import 済みのはず・確認すること）

### M-B（強く推奨）: 監査前に既定パスへ表を書いている
`dump(args.output)` が監査より先に走るため、リーク監査失敗時に汚染表が残る。**修正案 A**（推奨）: `args.output.with_suffix(".bin.tmp")` に書き、監査合格時のみ `os.replace()` で本番パスへ。最低でも失敗時に output を unlink。

### M-C（必須）: coverage 不一致の例外で manifest が 1 バイトも書かれない
`_coverage` の `raise RuntimeError(...)` が manifest 書き出しに到達しない。manifest 書き出しを `try/finally` に入れるか、coverage 不一致を例外でなく `coverage_pairs_match: false` + 非ゼロ終了コードで表現する（監査の扱いと同じ流儀）。

### M-D（強く推奨）: coverage 経路にテストが無い
合成 d1（train/test 各 2 行・`span_tokens` 混在）と合成 `coverage_pairs.jsonl` を `tmp_path` に置き、`d1_test_gold_pair_coverage` / `_span1_only` / `coverage_pairs_match` の期待値を直接 assert するテストを 1〜2 件追加。不一致で非ゼロ終了することの負テストも 1 件。

### C-1（対応）: manifest の扱い
現状の `data/corpus_freq_v1.manifest.json` は v1 の遺物（bin と不一致）。**修正後に本番ビルドを再実行して manifest を再生成する**。手書きで manifest を作らないこと。

## 本番ビルド再実行（M-A 修正後）

```bash
# スモーク（200 記事・高速確認）
uv run python scripts/build_corpus_freq.py --max-articles 200 \
  --output /tmp/cf-smoke.bin --manifest /tmp/cf-smoke.json

# 本番（100MiB 窓・10-30分）— バックグラウンドで回すこと（フォアグラウンドはタイムアウトする）
mkdir -p artifacts/issue19
nohup setsid uv run python scripts/build_corpus_freq.py \
  --output data/corpus_freq_v1.bin --manifest data/corpus_freq_v1.manifest.json \
  > artifacts/issue19/corpus-build.log 2>&1 < /dev/null &
echo $!  # PID を記録
```

**完了条件（本番ビルド後）**:
- manifest の 3 点一致: `utf8_bytes_scanned == 104852271` / `articles.seen == 5104` / `evaluation_overlap_skipped == 3`（d1 manifest と一致しなければ走査窓ズレ）
- `leak_audit.passed == true`・`counted_article_ids_intersecting_d1_dev_test == 0`
- `coverage.d1_test_gold_pair_coverage` が埋まっている（期待 ≈0.856・分岐 C 判定）
- bin の SHA-256 が再現するか確認（決定的 dump なので同じ入力なら `065bb103…` のはず。変わったら報告）
- `uv run pytest -q` が 654 + 新規 passed / 2 failed

## 制約

- `src/beme/lattice.py` / `converter.py` / `features.py` 変更禁止
- `src/beme/corpus_freq.py` は M-A/M-B に関わる変更のみ（API は変えない）
- 依存パッケージを増やさない
- `git commit` / `git push` しない（作業ツリーに残す）
- 日本語で報告。修正内容・ビルド結果・3 点一致・SHA-256・pytest 数を先頭に
