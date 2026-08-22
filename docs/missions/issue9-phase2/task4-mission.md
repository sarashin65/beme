# Task 4: 再現可能な学習 CLI

## 目的

train/dev/test を明示し、dev だけで epoch を選び、選択後に held-out test を評価して、学習済み weights と機械可読 report を決定的に保存するコマンドを追加する。

最初に `README.md`、`docs/missions/issue9-phase2/PLAN.md` §2.4、`scripts/evaluate.py`、Task 2–3 の API を読むこと。

## 依存

Task 1–3 が完了していること。

## 対象ファイル

- 新規: `scripts/train_scorer.py`
- 新規: `tests/test_train_scorer_cli.py`

training core、既存 scripts、辞書/weights/data は、CLI に不足する契約がない限り変更しない。不足があれば先に報告する。

## 変更内容

1. 必須/既定引数を PLAN §2.4 どおり実装する。正の epochs/rate と整数 seed を厳格検証する。
2. weight file の `feature_spec` から extractor を選ぶ。Phase 2 の既定は `dictionary_v2.tsv` + `decomposed_v1.json`。
3. `random.Random(seed + epoch).shuffle(indices)` により train のみを決定的に shuffle する。dev/test 順は固定する。
4. baseline dev/test と、各 epoch の train/dev、選択済み model の test を report する。test を epoch 選択には使わない。
5. report に全 CLI hyperparameter、git commit（取得できる場合）、Python/BeME version、辞書/初期 weights/split SHA-256、split 件数、選択 epoch、updates/averaging steps、全 error 指標を入れる。
6. weights/report は temp file を同一 directory に作って atomic replace。JSON は sort key・indent 2・UTF-8・末尾改行。
7. 決定的 artifact に日時、所要時間、絶対 path、ランダム temp 名を入れない。
8. 入力欠損・schema error・出力親 directory 不在/作成失敗は argparse error 相当の明確な非 0 終了にする。

## 必須テスト

- tiny synthetic split を subprocess 実行し weights/report を生成する。
- 同じ seed の独立 2 実行で両ファイルが byte-identical。
- 異なる seed が report metadata に反映される（結果変化そのものは強制しない）。
- report の split SHA/件数、baseline、selected epoch、test 指標を検証する。
- output weights を別 subprocess が `Weights.load` できる。
- missing split、不正 rate/epoch、feature spec mismatch が非 0。
- test 指標を変えた fixture でも selected epoch が変わらない。

## 完了条件

- PLAN 記載のコマンド形で学習できる。
- 同じ入力 bytes と引数から byte-identical artifact を生成する。
- dev/test 分離が report とテストで証明される。

## 検証コマンド

```bash
uv run pytest -q tests/test_train_scorer_cli.py
uv run python scripts/train_scorer.py --help
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- 対象外ファイルを変更しない。`data/d1` や本番 artifact を生成・コミットしない。
- test を見て hyperparameter/epoch を選ばない。
- ネットワーク・GPU・外部モデルを必要としない CLI/test にする。
- relevant test を追加し、コマンド実結果を報告する。

