# Task 8: アブレーション実行ハーネスと GPU マシンでの全実行

## 目的

Task 1〜7 の成果物を使い、PLAN §4 のアブレーションマトリクス 12 条件 × 3 seed = **36 回の学習**を GPU マシンで実行し、条件ごとの全指標を 1 つの機械可読な成果物へ集約する。McNemar 検定で条件間の差を判定する。

最初に `docs/missions/issue9-phase3/PLAN.md` の §3.8・§4・§6・§7、`scripts/train_scorer.py`、Task 1〜7 で追加された各スクリプトの `--help` を読むこと。

## 依存

Task 1〜7 がすべて完了していること。Task 2 または Task 3 が未完なら、該当条件（M0 / M1 / F0 / F1 / X0）を skip として記録し、残りを実行する。**未完の条件を推測値で埋めない。**

## 実行環境

**学習と d1 評価は GPU マシン `beme-gpu` で行う。** d1 の `*.jsonl` はそこにしか無い（`.gitignore:8` により repo 外）。

セットアップ:

1. GPU マシンに新しい作業ディレクトリ `~/beme-p3` を作り、**本ブランチを git clone する**（`~/beme-p2` は git 管理外の素の展開物で、`git rev-parse` が失敗するため `train_scorer.py` の `git_commit` が `null` になる。Phase 2 の run1 report がその状態だった）。
2. `~/beme-d1/repo/data/d1/{train,dev,test}.jsonl` を `~/beme-p3/data/d1/` へ配置する（コピーまたは symlink）。**miniPC へ持ち帰らない。**
3. `uv sync` して `uv run pytest -q` が全通しすることを確認してから実行に入る。

実測済みの所要時間: 学習 1 回 約 2 分（Phase 2 の run1/run2 の mtime 実測）、`--suite all` 評価 1 回 約 16 秒（miniPC 実測）。36 回を逐次で約 72 分、6 並列で約 15 分。

## 対象ファイル

- 新規: `scripts/run_ablations.py`
- 新規: `tests/test_run_ablations.py`
- 生成物: `artifacts/issue9-phase3/`（**repo にコミットしない。gitignore 対象**）

`src/beme/` 配下と Task 1〜7 のスクリプトは変更しない。不足があれば勝手に広げず報告する。

## 変更内容

### 1. 実行ハーネス（`scripts/run_ablations.py`）

条件表を JSON か Python の定数で持ち、各 (条件, seed) について次を順に実行する。

1. `scripts/train_scorer.py`（`--epochs 5 --learning-rate 0.1 --dictionary data/dictionary_v2.tsv`、`--initial-weights` は条件の `data/weights/ablations/<spec_id>.json`、`--seed` は 9 / 17 / 33）
2. `scripts/evaluate.py --suite all --reranker mock --dictionary data/dictionary_v2.tsv --weights <学習済み> --json`
3. `scripts/oracle_recall.py`（d1 test・Mozc・handmade、baseline と学習済みの両方）
4. `scripts/diagnose_cases.py`（d1 test の層別 + handmade / Mozc の順位表）
5. `scripts/measure_cost_scale.py`（baseline vs 学習済み）

**出力先は (条件, seed) ごとに別ディレクトリにすること。** `train_scorer.py` は出力先の親ディレクトリに `.beme-tmp-*` を作って atomic replace する（`train_scorer.py:110-127`）ので、並列実行で同一ディレクトリへ複数プロセスを向けると衝突する。

推奨レイアウト:

```text
artifacts/issue9-phase3/<condition_id>/seed<N>/{weights.json,report.json,evaluate.json,oracle/*.json,diagnose/*.json,cost_scale.json}
```

- `--conditions` で条件を絞れること（Task 2/3 未完時の部分実行、失敗した条件の再実行に使う）。
- `--seeds` で seed を絞れること。
- `--jobs` で並列度を指定できること。
- **既存の出力ディレクトリがある場合は上書きせずエラー**（`--force` で明示的に上書き）。長時間の実行結果を事故で消さないため。
- 途中で失敗した条件は記録して続行し、最後に失敗一覧を非 0 終了で報告する。1 条件の失敗で 36 回を全部やり直さない。

### 2. A0 ゲート（必須・最初に実行する）

**A0（`decomposed_v1` 無改造）を最初に seed 9 で実行し、Phase 2 の値を再現することを確認してから残りへ進む。**

| 指標 | Phase 2 の値 | 許容 |
|---|---:|---|
| d1 test model errors | 538（26.9%） | ±1 例 |
| d1 test exact matches | 1,462（73.1%） | ±1 例 |
| d1 test coverage errors | 0 | 完全一致 |
| baseline d1 test model errors | 620（31.0%） | 完全一致 |
| 選択 epoch | 4 | 一致が望ましい（不一致なら報告） |
| AJIMEE CER | 0.287092 | 完全一致 |
| Mozc | 87/498 | 完全一致 |
| handmade Top-1 | 13/14 | 完全一致 |

**再現しなければそこで止めて報告すること。** 残り 35 回を回しても比較対象が無い。第一の容疑者は commit 差（`~/beme-p2` は git 管理外だった）、次が Task 1 の特徴フィルタが `decomposed_v1` の出力を変えてしまっていること。

### 3. 集約と McNemar 検定

全条件の実行後、`artifacts/issue9-phase3/summary.json` を作る。

- 条件ごとに 3 seed の **中央値と最小–最大幅**（Phase 2 の dev エラーは epoch 間で 1〜2 例しか動かず、単一の値はノイズと区別できない）。
- **A0 との McNemar 検定**: d1 test 2,000 件は対応ありデータなので、正誤の 2×2 分割表 `(A0 正/誤) × (条件 正/誤)` を作り、不一致セル `b`, `c` に対して検定する。`b + c < 25` のときは二項検定の厳密版、それ以上は連続性補正つき χ² を使い、**どちらを使ったかを出力に記録する**。
  - 分割表は代表 seed（9）で作る。3 seed 分の分割表も併記する。
  - 標準ライブラリ（`math`・`statistics`）で実装する。`scipy` を新規依存に加えない。
- **判定ラベルは 3 値に固定する**。
  - `significant_worse`: p < 0.05 かつ中央値が悪化 → その特徴族は必要
  - `significant_better`: p < 0.05 かつ中央値が改善 → その特徴族は有害または冗長
  - `indeterminate`: p ≥ 0.05 → **「差が無い」ではなく「2,000 件では判別できない」**
- 既存 suite（AJIMEE 200 / Mozc 498 / handmade 14）の値も載せるが、**採否判断には使わない**。件数が小さく検出力が無い（handmade は 1 件で Top-1 が 7.1 pt 動く）。この但し書きを機械可読なフィールドとして summary に入れること。
- 学習済み重みは、生値に加えて **`lex.ipadic_cost_mapped` を 1 とした正規化値**を併記する。理由は下記。

### 4. 重みの解釈に関する必須の注記

PLAN §2.2(b) で実測確認済みの構造的事実:

```text
任意の完全経路について  Σ lex.script.*  ==  |P|  ==  Σ trans.*
```

（Mozc 200 読み × K-best 5 = 1,000 経路で照合し不一致 0）

つまり特徴空間に**厳密な線形従属が 1 本**あり、「トークン数ペナルティ」方向の重みは個別に同定できない。Phase 2 の学習結果でも `lex.script.*` が 0 → 1.02〜1.39、`trans.default` が 0.12 → 1.03 と同時に増えている。

**summary.json に、生の重み値を特徴の重要度として読んではならない旨を機械可読なフィールドとして入れること。** 採否判断は必ずエラー率差（McNemar）で行う。

同様に `lex.reading_len` は `Σ len(token.reading) == len(reading)` により gold−pred が常に 0 で、**構造的に学習されない**（Phase 2 の学習済み重みに項が存在しない）。全条件で 0 のままであることを確認し、summary に記録すること。

### 5. A2（transition only）の縮退の扱い

`lex.*` を全除去した条件では、`trans.bos = 0.0` のため 1 トークン経路のコストが厳密に 0 になり、大量の同点を `_path_key` のタイブレークが決める。miniPC での実測例: `はしのうえ → 橋之上`（1 トークン・cost 0.000）。

**これはバグではなくアブレーションの正しい下限である。** 例外にせず、`tie_count`（1-best と同コストの経路数）を記録して報告すること。

## 必須テスト

- 条件表の全 spec id が Task 1 のレジストリに存在する。
- 出力ディレクトリの衝突検出（既存ディレクトリで `--force` 無しならエラー）。
- 1 条件の失敗が全体を止めず、失敗一覧に載って非 0 終了する。
- McNemar 検定が既知の分割表で手計算値と一致（厳密版・χ² 版の両方）。
- `b + c` の閾値で検定手法が切り替わり、どちらを使ったかが記録される。
- A0 ゲートの許容判定が、閾値ちょうどの値で意図どおり pass / fail する。
- 中央値と最小–最大幅の計算が 3 seed で正しい。
- summary の解釈注記フィールドが存在する。
- `--conditions` / `--seeds` による部分実行。
- テストは実データ無しの合成 fixture（学習を実際に回さないダミー成果物）で完結すること。**テストで 36 回の学習を回さない。**

## 完了条件

- A0 ゲートを通過している。
- 12 条件（Task 2/3 未完なら該当条件を skip 記録）× 3 seed の成果物が揃っている。
- `summary.json` に全条件の指標・McNemar 判定・解釈注記が入っている。
- 失敗・skip した条件が明示的に記録されている。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
# miniPC（ハーネスの単体検証。学習は回さない）
uv run pytest -q tests/test_run_ablations.py
uv run python scripts/run_ablations.py --help
uv run pytest -q

# GPU マシン（~/beme-p3）
uv run pytest -q

# A0 ゲート（これが通るまで先へ進まない）
uv run python scripts/run_ablations.py --conditions A0 --seeds 9 \
  --output-root artifacts/issue9-phase3

# 残り全条件
uv run python scripts/run_ablations.py --jobs 6 \
  --output-root artifacts/issue9-phase3

# 集約
uv run python scripts/run_ablations.py --summarize-only \
  --output-root artifacts/issue9-phase3
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- `artifacts/issue9-phase3/` を repo にコミットしない。`.gitignore` に入っていなければ追加を報告する（**`.gitignore` の編集は許可する**）。
- `src/beme/` 配下と Task 1〜7 のスクリプトを変更しない。不足があれば報告する。
- `data/d1/*.jsonl` を miniPC へコピーしない。GPU マシン上でのみ扱う。
- `scipy` などの新規依存を追加しない。McNemar は標準ライブラリで実装する。
- **test を見てハイパーパラメータや epoch を選ばない。** epoch 選択は `train_scorer.py` が dev のみで行う既存ロジックに任せる。
- Phase 2 の `~/beme-p2/artifacts/` の学習済み重みを再利用しない（metadata の `updates` / `averaging_steps` が修正前の末尾 epoch 値のまま。Phase 2 コメントの引き継ぎ注意）。A0 として学習し直す。
- 実行できなかった条件を推測値で埋めない。
- 上記コマンドの実結果（A0 ゲートの実測値を含む）を報告する。
