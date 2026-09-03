# task2 — S 実測: 最適化要因の排除（lr / epoch スイープ）

対象 issue: #19（S 項目・着手順序表 2 行目）
実行機: GPU マシン `ssh ws`（sarashin@192.168.0.28）
依存: なし（task1 と並行可・task3 以降を**ブロックしない**）
種別: 実行のみ（**コード変更ゼロ**）

---

## 0. 目的

issue #19 の主張は「train 誤差 28.2%（平均化重み）が下がらないのは**容量**の不足であって、**最適化**の失敗ではない」である。この主張はまだ実測されていない。学習率が大きすぎて発散気味なだけ、エポックが足りないだけ、という説明を潰す。

これが潰れないうちは、F2 の結果（効いた / 効かなかった）をモデル容量の話として解釈できない。

---

## 1. 必ず読むもの

- `gh issue view 19`（本文 + コメント §1「平均化重みでの train 評価: 誤差 28.2%」）
- `docs/missions/issue19/PLAN.md` §1（タスク表）・§5（共通制約）
- `src/beme/training.py`（`train_scorer` / `evaluate_split` / `evaluate_test`。lr がどう効くか・平均化重みがいつ作られるか）
- `scripts/train_scorer.py`（CLI。`--learning-rate` / `--epochs` は既に引数になっている）
- `scripts/run_ablations.py:216`（既存の学習コマンド 1 本の完全な形）

---

## 2. やること

### 2.1 スイープ設計（事前登録）

base は **A8 = `decomposed_v1-no_script-surface_len`**（現行 base・`data/weights/ablations/decomposed_v1-no_script-surface_len.json`）。

| 軸 | 値 |
|---|---|
| learning rate | 0.01 / 0.03 / **0.1（現行）** / 0.3 / 1.0 |
| epochs | **5（現行）** と 20 |
| seed | 9（lr 軸）・lr 最良点のみ 9/17/33 |

実行数: lr 5 点 × epochs 2 = 10 run（seed 9）+ 最良点の追加 2 seed = 12 run。

### 2.2 train 誤差は平均化重みで測る

現行の report.json の `epochs[].train` は**学習中のオンライン誤差（生の重み）**であり、dev/test は平均化重みである（issue #19 の注意書きと、Hermes の検証が確認済み）。

各 run の**最終平均化重み**で `data/d1/train.jsonl` を評価し直し、その誤差を記録すること。既存 CLI にその出力は無いので、次のどちらかで取る（**`src/beme/` は変更しない**）:

```bash
# 学習済み重みで train split を評価する（推論のみ・CE 不要）
uv run python scripts/evaluate.py --suite ... # ← 使えない。d1 split は suite ではない
```

→ 正しいやり方: **`scripts/diagnose_cases.py --data data/d1/train.jsonl`** を、`--baseline-weights` に A8 初期重み・`--learned-weights` に各 run の学習済み重みを渡して回す。出力の `learned.end_to_end_error_rate`（全体）が平均化重みでの train 誤差である。層別は要らないので `--stratify` は付けなくてよい。

これが使えない/意味がずれると判断したら、**その旨を報告に書いて止まる**（新しい評価スクリプトを勝手に足さない）。

### 2.3 判定基準（事前登録）

| 観測 | 結論 |
|---|---|
| どの lr・どの epochs でも平均化重みでの train 誤差が **26〜30% の帯**に収まる | **最適化要因は排除された。28% は容量由来**。issue #19 の主張が確定する |
| いずれかの設定で train 誤差が **24% を下回る** | 最適化が律速していた可能性がある。issue #19 の「過少適合」の解釈は維持されるが、**その設定を F2 の学習設定に採用すべき**。task5 に申し送る |
| 大きい lr で train 誤差が悪化して発散する | 想定内。現行 lr 0.1 が発散側でないことの確認になる |

---

## 3. 実行手順

1. `ssh ws` が通ることを確認する。**通らなければこのタスクは「未実施」として報告し、終わる**（task3 以降はこのタスクを待たない）
2. この worktree から git bundle を作って転送する（プライベート repo・GPU 側に gh 認証が無い。issue #17 Phase 4.1 と同じ手順）
   ```bash
   git bundle create /tmp/beme-19.bundle feat/issue19-f2
   scp /tmp/beme-19.bundle ws:/tmp/
   ssh ws 'git clone --branch feat/issue19-f2 /tmp/beme-19.bundle ~/beme-19 && cd ~/beme-19 && uv sync'
   ```
3. d1 を供給する（GPU 側の既存 d1 を symlink: `ln -s ~/beme-d1/repo/data/d1 data/d1`）。**供給後に SHA-256 が `data/d1/manifest.json` の `artifacts_sha256` と一致することを確認する**
4. 12 run を回す。1 run ずつ `scripts/train_scorer.py` を直接叩く（`run_ablations.py` は条件×seed の形なので lr 軸には合わない）
   ```bash
   uv run python scripts/train_scorer.py \
     --train data/d1/train.jsonl --dev data/d1/dev.jsonl --test data/d1/test.jsonl \
     --dictionary data/dictionary_v2.tsv \
     --initial-weights data/weights/ablations/decomposed_v1-no_script-surface_len.json \
     --epochs 5 --learning-rate 0.1 --seed 9 \
     --output-weights artifacts/issue19/lr-sweep/lr0.1_ep5_s9/weights.json \
     --output-report  artifacts/issue19/lr-sweep/lr0.1_ep5_s9/report.json
   ```
5. 各 run について §2.2 の方法で平均化重みでの train 誤差を測る
6. 長引くならデタッチ起動（`nohup setsid ... < /dev/null &`・PPID=1 を確認・完了は `while kill -0 <PID>; do sleep 30; done`）
7. artifacts をローカルへ回収（`scp -r ws:~/beme-19/artifacts/issue19/lr-sweep artifacts/issue19/`）

---

## 4. レポート

`docs/missions/issue19/lr-sweep-report.md`。

- 先頭に結論（§2.3 の 3 行のどれに該当したか）
- 12 run の表: lr / epochs / seed / オンライン train 誤差(ep1, 最終) / **平均化重み train 誤差** / dev 誤差 / test 誤差
- 「28% は容量由来」が確定したか否かの明示的な判定
- F2 の学習設定への申し送り（lr を変えるべきか否か）
- 実行機名・所要時間・artifact SHA-256

---

## 5. 完了条件

- [ ] 12 run が完走し、`artifacts/issue19/lr-sweep/*/report.json` が揃っている（またはこのタスクが「`ws` 未復旧のため未実施」として明示的に報告されている）
- [ ] 各 run の平均化重みでの train 誤差が測られている
- [ ] `docs/missions/issue19/lr-sweep-report.md` に §2.3 の判定が書かれている
- [ ] `git diff --stat origin/main` が **`docs/` 以外空**（このタスクはコードを変更しない）
- [ ] git commit / push をしていない

---

## 6. 検証コマンド

```bash
cd /home/sarashin/work/beme-19

# コード無変更の確認（これが最重要）
git diff --stat origin/main -- src/ scripts/ tests/    # 空であること
git status --short

# 回収した artifact の一覧と主要数値
uv run python -c "
import json,glob
for p in sorted(glob.glob('artifacts/issue19/lr-sweep/*/report.json')):
    d=json.load(open(p))
    print(p, d['cli']['learning_rate'], d['cli']['epochs'], d['cli']['seed'],
          [round(e['train']['end_to_end_error_rate'],4) for e in d['epochs']],
          round(d['selected_model']['test']['end_to_end_error_rate'],4))
"
```

---

## 7. 制約（厳守）

- **コードを一切変更しない。** `src/` も `scripts/` も `tests/` も触らない。既存 CLI の引数だけで完結する設計になっている
- **`src/beme/lattice.py` / `src/beme/converter.py` 変更禁止**（探索コア）
- `git commit` / `git push` をしない
- 作業はこの worktree（と GPU 側の `~/beme-19`）の中だけ。`git stash` を使わない
- **このタスクは task3-6 をブロックしない。** `ws` が復旧しなければ「未実施」で終えてよい。その場合 task7 のレポートに「最適化要因の排除は未実施であり、容量由来という解釈は未確定である」と書けるよう、報告に明記する
- 判定基準（§2.3）を事後に動かさない
- 日本語で報告。数値と結論を先頭に
