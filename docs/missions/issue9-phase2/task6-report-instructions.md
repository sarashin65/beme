# Task 6 指示: README 追記 + phase2-report.md 作成（実測値は下記のとおり）

Task 6 の最終レポート作成です。**あなたは実測値の計測をやり直す必要はありません**。下記の実測値をそのまま使い、`docs/missions/issue9-phase2/phase2-report.md` を作成し、README に Phase 2 の学習・再現・評価コマンドを追記してください。

## 対象ファイル

- `README.md`: Phase 2 学習・再現・評価コマンド、データ前提を追記
- 新規: `docs/missions/issue9-phase2/phase2-report.md`
- 必要なら `.gitignore`: 学習 artifact directory（`artifacts/`）だけを限定追加

製品コード、tests、データ JSONL、辞書、既存 weights は変更しない。git commit / push 禁止。

## 実測値（GPU マシン・`~/beme-p2`・2026-08-22 21:0x JST 実行）

### 学習コマンド（固定パラメータ）

```bash
uv run python scripts/train_scorer.py \
  --train data/d1/train.jsonl --dev data/d1/dev.jsonl --test data/d1/test.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --initial-weights data/weights/decomposed_v1.json \
  --epochs 5 --learning-rate 0.1 --seed 9 \
  --output-weights artifacts/issue9-phase2/run1/weights.json \
  --output-report artifacts/issue9-phase2/run1/report.json
```

- **再現性**: 同一コマンド 2 回（run1/run2）の weights.json / report.json が **byte-identical（cmp 一致）**
- **selected_epoch: 4**（dev の (end_to_end_error_rate, model_error_rate, epoch) 辞書順最小で選択・test は選択後に 1 回だけ評価）

### データ（d1・#5 作成・provenance 記録済み）

- train 10,000 / dev 1,000 / test 2,000・記事 disjoint
- manifest 検証: `article_disjoint_verified=true` / `article_sha256_consistent=true` / `contrast_pairs_disjoint=true` / `split_field_verified=true`
- span_tokens は 1〜3（int）・source は Wikipedia 20231101.ja・MeCab かな化由来
- d1 は #5（IME 専用小型 LM 学習）用に作成されたデータで、BeME の学習データとしては初利用

### 学習結果（held-out test 2,000 件）

| 指標 | baseline（decomposed_v1） | learned（averaged・epoch 4） |
|---|---:|---:|
| model error rate | 31.0%（620 件） | **26.9%（538 件）** |
| exact match | 69.0%（1,380 件） | **73.1%（1,462 件）** |
| coverage error | 0 | 0 |

train 内訳（全 epoch・coverage 0）: epoch1 exact 6,770 / epoch2 6,752 / epoch3 6,694 / epoch4 6,699 / epoch5 6,665

### 既存評価 suite（learned weights・別プロセス `scripts/evaluate.py`）

| suite | baseline（decomposed_v1） | learned |
|---|---|---:|
| AJIMEE CER | 0.290080 | **0.287092**（微改善） |
| AJIMEE exact | 0.070 | **0.080** |
| Mozc | 88/498 | 87/498（1 件減） |
| Handmade NDCG@5 | 0.992972 | 0.961061 |
| Handmade Top-1 | 1.000（14/14） | 0.928571（13/14） |

### handmade ケース単位比較（baseline → learned・14 件）

baseline は 14 件全て Top-1 正解。learned で **1 件のみ回帰**:

- **hashi-no（はしの）**: baseline Top-1「橋の」✅ → learned Top-1「橋野」❌（2 位に「橋の」）
  - 考察: d1（Wikipedia 由来）で固有名詞「橋野」が高頻度・学習重みが人名/地名表記を好む方向に動いたためと推測。「の」助詞接続の境界選択が影響
- 他の 13 件は Top-1 維持（kisha/hashi/ame/kaeru/koushou/kyoukai/seika/taishou/kisha-wa/koushou-wo/kyoukai-de/koushou-history/koushou-business）
- 候補順位は複数件で変化（例: kisha の 5 位が「帰社」→「騎射」等）・NDCG は一部変動

### その他

- 学習済み weights は別プロセスの `scripts/evaluate.py --weights <path> --suite all --reranker mock` で読み込み成功（上記 suite 結果がその証跡）
- full pytest: **511 passed**（学習後も全既存テスト green）
- learned cost scale の変化により confidence/cascade/reranker blend は変動する → Phase 3 で再校正が必要（PLAN §6）

## phase2-report.md 必須項目（PLAN §Task 6 指示から）

- commit（現在の HEAD: 9b0510b 以降の Task 6 作業含む・`git log -1 --format=%H` で取得可）、辞書/初期 weights/split SHA-256、全 CLI 引数
- split 件数、provenance、article disjoint、MeCab かな化と Wikipedia 分布の注意
- epoch ごとの train/dev 指標、dev 選択規則と選択 epoch（4）
- held-out test の baseline vs learned（coverage/model/end-to-end）
- 学習済み weights の path/hash/metadata と別プロセス再現結果
- handmade 14 件の case-by-case before/after（上記表）、AJIMEE、Mozc
- 失敗/coverage 例の代表サンプル（coverage 0 なので model error 例を 2-3 件・personal info なし）
- Phase 3 引き継ぎ（ablation、cost scale 再校正、oracle recall@k、data bias、配布）

## README 追記内容

- Phase 2 の学習コマンド（上記固定例）・再現コマンド（cmp 検証）・評価コマンド（--weights）
- データ前提: `data/d1/*.jsonl` が必要（gitignore・repo 外供給・GPU マシン `~/beme-d1/repo/data/d1/` に存在）

## 検証

- `uv run pytest -q` が通ること（レポート作成でコードを壊さない）
- README/レポートの記述が上記実測値と一致すること
- 完了報告に作成ファイル一覧と commit 予定内容を書く
