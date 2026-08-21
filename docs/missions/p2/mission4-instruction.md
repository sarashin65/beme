# Mission 4: 文全体ラティス化（スパン境界排除）による P2 実現

## 指示

`docs/missions/p2/PLAN.md` の **Mission 4** を実装してください。Mission 1-3（LatticePath 永続化・Backward DP・A* Exact K-best）は完了済みです。

## 背景（現状の問題）

現在の `convert_sentence`（`src/beme/benchmark.py`）は、非かな文字（句読点・記号・数字）で入力文を**スパン分割**し、各スパンごとに `converter.convert(k=5)` を呼んで、各スパンのトップ 5 同士だけを合成して再びトップ 5 を取る**局所 Pruning** を行っている。これにより正解パスが早期に脱落する（multi-span の完全一致率 3.8% が single-span 10.5% を大きく下回る原因）。

## 実装内容

1. **非かな文字も Lattice に含める**: `build_lattice`（`src/beme/lattice.py`）を拡張し、非かな文字・記号を「コスト 0・1 文字確定ノード」の固定エッジとして Lattice に組み込む
2. **文全体を単一 Lattice として処理**: `convert_sentence`（`src/beme/benchmark.py`）の「スパン分割 → 各スパン convert(k=5) → 合成」の局所 Pruning を廃止し、文全体を 1 回の A* K-best で処理する
3. スパン境界の接続コストも自然に計算されるようにする

## 完了条件

- `scripts/evaluate.py --suite ajimee --json` で:
  - CER が悪化しない（現状 0.290080 前後）または改善する
  - 完全一致率が悪化しない（現状 14/200 = 7.0%）または改善する
- `scripts/evaluate.py --suite mozc --json` で pass 数が悪化しない（現状 88/498）
- `scripts/evaluate.py --suite handmade --json` で完全一致が維持される（現状 Top-1 1.0）
- 既存テスト全件パス（`uv run pytest -q`）
- 評価結果を報告に含める（変更前後の数値比較）

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- 実装後は自己確認（評価実行・テスト実行）して報告
- **評価が改善しない場合も、数値を正直に報告する**（悪化したならその理由の考察を添える）
