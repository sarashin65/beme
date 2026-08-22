# BeME D1 評価レポート（2026-08-22）

## 1. 結論

**D1 は aggregate な decoder signal を示したが、PLAN-v2 §8 の decoder signal Go は総合 No-Go と判定する。D2 / S200 の品質比較へはまだ進まない。**

- locked test conditional Top-1 は既存順位 69.00% から D1 75.20% へ **+6.20 points**。paired bootstrap 95% CI は **[+3.80, +8.60] points**であり、「+3 points 以上かつ CI 下限 > 0」は通過した。
- しかし、文脈対照 64 pair の両方正解率は **37.50%**で、指定基準 80% に未達だった。
- baseline-correct regression は **18.41%**で、PLAN-v2 の 5% 以下に未達だった。
- high-frequency distractor subset は 65.89%で全体 75.20%より **9.31 points 低く**、許容差 5 points に未達だった。
- BeME 統合評価は AJIMEE と Mozc を改善した一方、handmade を 14/14 から **8/14**へ悪化させた。scratch / FT 採用 hard gate を満たさない。

従って、今回の結果は「task SFT に有意な ranking signal がある」証拠ではあるが、「次の model arm へ進める品質 gate を通った」証拠ではない。

## 2. 評価対象と再現情報

### 2.1 学習済みモデル

| 項目 | 観測値 |
|---|---:|
| arm | D1: Qwen2.5-0.5B base + QLoRA task SFT |
| base revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| seed | 20260822 |
| train / dev | 10,000 / 1,000 groups |
| epoch / step | 1.0 / 625 |
| objective | answer surface token + EOS の causal LM loss |
| max sequence | 128 |
| micro-batch / gradient accumulation | 1 / 16 |
| learning rate | 2e-4 |
| LoRA | rank 16, alpha 32, dropout 0.05; q/k/v/o/up/down/gate projection |
| quantization / compute | NF4 double quantization / BF16 |
| trainable parameters | 8,798,208 |
| train loss | 1.142846 |
| eval loss | 2.020775 から 1.709414 へ改善 |
| train runtime | 1,557.81 s |
| training peak allocated / reserved | 1,356,700,160 / 2,871,001,088 bytes（reserved 2.674 GiB） |
| resume | checkpoint-250 から再開し step 625 まで完走 |

`training_log.jsonl` と `resume.log` に NaN / OOM / out-of-memory の記録はなかった。

### 2.2 artifact

| artifact | path / SHA-256 |
|---|---|
| adapter | GPU `~/beme-d1/runs/d1-seed-20260822/final_adapter/adapter_model.safetensors` / `d18ac67d52d662222b15fa55708f6722c0ac96b8137c135ab296c6441b7b2612` |
| adapter config | GPU `.../final_adapter/adapter_config.json` / `eb6ecbb468ff34f713d74669d7373f6969db9208e2bbd9d16ca7472e0ad00c91` |
| locked test | `data/d1/test.jsonl` / `7b0ad8ff3f9abf11ccfb9488214db0199ecd9b06d07928dea0fb6c8681f6f2ff` |
| candidate scores | GPU `~/beme-d1/artifacts/d1-candidate-scores-20260822.jsonl` / `9671fc313075b34d7dcd35f4c0cc5d8adc8ff6f8b6e00b5cf1d00b396531e06b` |
| integrated suites | GPU `~/beme-d1/artifacts/d1-suites-20260822.json` / `eb3305da899e65605d53565969e2837a1020205b9a5caea59d2c3e63bfd7832b` |
| final evaluator working tree | `scripts/evaluate_d1.py` / `96387258133d408b33b4305f0f28620484990ac4db13f64d19850925f9cbe44c` |

Git commit / push は miniPC、GPU machine とも行っていない。

## 3. 評価 protocol

### 3.1 conditional ranking

- 主条件は `left_context + reading + right_context`。補助条件として left-only も測定した。
- 各候補について answer token と EOS の条件付き log probability を計算した。
- `sum` と token `mean` のどちらを使うかは dev のみで選択した。両条件とも `sum` が選ばれた。
- test は 2,000 groups、98,402 candidates。全 group で gold-in-candidate のため conditional ranking の分母は 2,000。
- 比較対象は BeME の mock / 既存順位。locked test 上で literal `MockContextReranker` を実行し、既存 Top-1 から変わった group は 0/2,000、Top-1 は 69.00%だった。従ってこの test では mock と Viterbi 既存順位が同一である。
- CI は同じ test item を再標本化する paired percentile bootstrap、10,000 resamples、seed 20260822。差は `D1 correct - baseline correct`。

### 3.2 文脈対照

manifest の `test.contrast_pairs` は 64。各 pair は同一 reading に対する異なる文脈の 2 item、合計128 itemである。PLAN-v2 の指標に合わせ、主値を「pair の両 item が Top-1 正解」とした。128 item 個別の Top-1 も併記した。

### 3.3 BeME 統合

`scripts/evaluate_d1.py` に `ConditionalReranker` を実装した。これは BeME `Reranker` Protocol の `rerank(candidates, context)` を満たし、candidate token 列から reading を復元し、lower-is-better の `Candidate.score` に conditional NLL を設定する。評価 adapter は Mozc の空文脈でも D1 を実際に呼ぶ。評価関数は既存の `evaluate`、`evaluate_ajimee`、`evaluate_mozc`、Top-5、rerank K=64 をそのまま使用した。Viterbi blend は decoder 単体測定のため使用していない。

推論は GPU machine 上で直接実行した。理由は、学習時と同じ pinned base revision、PEFT adapter、bitsandbytes 4-bit stack をそのまま利用でき、model RPC を追加せず評価 protocol の差を最小化できるためである。miniPC では current mock baseline を `uv run python scripts/evaluate.py --suite all --reranker mock` で fresh 測定した。GPU latency は production の N100 latency と比較できない。

AJIMEE の長文候補は training/locked-test 用 128 token 上限に収まらないため、suite inference の上限は 1,024 token とした。最初の combined run は conditional 6,000 records（dev/test × 2 context modes）を書き終えた後、AJIMEE の 128 token 制限で停止した。candidate artifact は 6,000 unique `(context_mode, id)` を確認してから集計した。suite は Protocol 実装で分離再実行し、evaluator の既定値も locked test 128 / suite 1,024 に分離した。

## 4. conditional Top-1 結果

### 4.1 dev での score 選択

| context | ranker | Top-1 | baseline 差 | 選択 |
|---|---|---:|---:|---|
| left+right | 既存順位 | 73.90% | — | — |
| left+right | D1 sum | **78.50%** | **+4.60 pt** | yes |
| left+right | D1 mean | 69.80% | -4.10 pt | no |
| left-only | 既存順位 | 73.90% | — | — |
| left-only | D1 sum | **77.10%** | **+3.20 pt** | yes |
| left-only | D1 mean | 66.00% | -7.90 pt | no |

### 4.2 locked test

| context | ranker | Top-1 | MRR | NDCG | baseline 差 | paired 95% CI |
|---|---|---:|---:|---:|---:|---:|
| left+right | mock / 既存順位 | 69.00% | 0.795621 | 0.844584 | — | — |
| left+right | **D1 sum** | **75.20%** | **0.845170** | **0.883210** | **+6.20 pt** | **[+3.80, +8.60] pt** |
| left-only | mock / 既存順位 | 69.00% | 0.795621 | 0.844584 | — | — |
| left-only | **D1 sum** | **73.65%** | **0.831747** | **0.872657** | **+4.65 pt** | **[+2.20, +7.05] pt** |

主条件 left+right では、baseline error 620件中378件を回復した一方、baseline correct 1,380件中254件を壊した。

| diagnostic | D1 left+right | gate | 判定 |
|---|---:|---:|---|
| recoverable-error recovery | 378/620 = 60.97% | 補助値 | — |
| baseline-correct regression | 254/1,380 = **18.41%** | 5%以下 | **Fail** |
| high-frequency distractor subset | 284/431 = **65.89%** | 全体比 -5 pt 以内 | **Fail（全体比 -9.31 pt）** |

## 5. 文脈対照 64 pair

| 条件 | 既存順位 | D1 sum | 差 | gate |
|---|---:|---:|---:|---|
| 128 item 個別 Top-1 | 38/128 = 29.69% | 79/128 = **61.72%** | +32.03 pt | 補助値 |
| 64 pair 両方正解 | 0/64 = 0.00% | 24/64 = **37.50%** | +37.50 pt | **80%以上: Fail** |

文脈利用は明確に改善したが、対照 pair の過半数をまだ両立できていない。「文脈を変えると選択を変えられる」signal と「80%の安定した文脈識別」は分けて判断する。

## 6. BeME 統合評価

### 6.1 quality

| suite | current mock baseline | D1 conditional reranker | 差 | hard gate |
|---|---:|---:|---:|---|
| handmade Top-1 | **14/14 = 100%** | **8/14 = 57.14%** | -42.86 pt | **14/14: Fail** |
| handmade NDCG@5 | 0.992972 | 0.583216 | -0.409756 | Fail |
| handmade MRR@5 | 1.000000 | 0.680952 | -0.319048 | Fail |
| AJIMEE mean CER | 0.290080 | **0.231293** | **-0.058788** | Pass（低いほど良い） |
| AJIMEE exact | 7.00% | **17.00%** | **+10.00 pt** | Pass |
| Mozc | 88/498 = 17.67% | **109/498 = 21.89%** | **+21件 / +4.22 pt** | Pass |

Paired bootstrap 95% CI（10,000 resamples、seed 20260822）:

- AJIMEE CER 差 `D1 - baseline`: **[-0.08530, -0.03229]**。0を跨がず改善。
- AJIMEE exact 差: **[+4.50, +15.50] points**。
- Mozc pass rate 差: **[+1.00, +7.43] points**。

AJIMEE の具体例:

- 改善: `ajimee-jwtd-v2-1589` は baseline「（政策曲」CER 0.75 から D1「（制作局」CER 0.00。
- 改善: `ajimee-jwtd-v2-1189` は baseline「千了知にいた」CER 0.50 から D1「占領地にいた」CER 0.00。
- 悪化: `ajimee-jwtd-v2-1919` は baseline「「笑いの野球」CER 0.00 から D1「「わらいのやきゅう」CER 1.00。

handmade の D1 Top-1 failure は `kisha→キシャ`、`hashi→ハシ`、`ame→アメ`、`seika→せいか`、`kisha-wa→キシャは`、`koushou-wo→考証を` の6件。とくに無文脈 sentinel で片仮名・かな表記へ寄る regression が観測された。原因が SFT data balance、sum score、または base prior のどれかは、この評価だけでは確定しない。

### 6.2 GPU inference telemetry

| 項目 | 観測値 |
|---|---:|
| scored candidates | 38,200 |
| model scoring wall time | 165.465 s |
| mean candidate latency | 4.332 ms / candidate |
| peak allocated | 2,398,347,776 bytes（2.234 GiB） |
| peak reserved | 7,438,598,144 bytes（6.928 GiB） |

これは RTX 3060 Ti、batch 16、最大1,024 token の suite 測定であり、N100 production latency gate の測定ではない。候補単位平均を変換全体 p95 と解釈しない。

## 7. PLAN-v2 §8 Go / No-Go

| gate | 観測 | 判定 |
|---|---|---|
| data: audit 98%以上 | 199/200 = 99.5% | Go |
| data: article-disjoint / overlap除去 | train/dev/test unique articles 696/65/129、評価 overlap 3記事除外、split manifestあり | Go |
| data: train 10k / manifest | train 10,000、source revision/license/artifact SHAあり | Go |
| decoder: Top-1 +3 pt、CI下限>0 | test +6.20 pt、CI下限 +3.80 pt。dev も +4.60 pt | Go |
| decoder: context contrast 80%以上 | pair 両方正解 37.50% | **No-Go** |
| decoder: regression 5%以下 | 18.41% | **No-Go** |
| decoder: frequency bucket崩壊なし | high-frequency distractor が全体比 -9.31 pt | **No-Go** |
| decoder: 8 GB安定 / resume | training reserved 2.674 GiB、suite reserved 6.928 GiB、resume完走、NaN/OOM記録なし | Go |
| scratch / FT: 3 seed同方向 | D1 1 seedのみ | 未判定 |
| scratch / FT: handmade 14/14 | D1 8/14 | **No-Go** |
| scratch / FT: Mozc 88/498以上 | D1 109/498 | Go |
| scratch / FT: AJIMEE非悪化 | CER paired CI 全域で改善 | Go |
| production: N100 warm p95 50 ms以下 | 未測定 | 未判定 |
| Colab: 8条件 | S200 scaling/corpus/runtime/予算等が未測定 | No-Go |

**総合判定: data Go、aggregate Top-1 signal Go、decoder signal gate は No-Go、scratch / FT 採用は No-Go / 未判定、production と Colab は未判定 / No-Go。**

PLAN-v2 の M1 文言は dev で D0 と Viterbi の双方との比較を求める。今回の依頼範囲は mock / 既存順位との比較であり、D0 は測定していない。従って「D0 を含む M1 全条件を通過」とは主張しない。

## 8. 次ステップ推奨

1. **D2、S200 quality run、Colab/S500 へは進まない。** loss と aggregate Top-1 だけを根拠に model/corpusを大きくすると、18.41% regression、文脈対照未達、handmade崩壊を残したまま計算量を増やす。
2. 次の run は D1 task adaptation の修正に限定する。test を tuning に使わず、dev の baseline-correct regression、contrast pair、high-frequency distractor、無文脈 sentinel を gate にする。新しい candidate sampling / score設定が必要なら train/dev artifact を作り、locked test は最後まで再参照しない。
3. decoder 単体 signal は確認できたため、dev 上で Viterbi blend weight、sum/length penalty、no-context 条件を校正する価値はある。ただし handmade だけの特例は入れず、baseline-correct regression 5%以下を目的にする。
4. 文脈対照は pair 両方正解 80%以上まで task data / objective を診断する。片側だけ当たる pair を context方向、frequency bucket、surface表記で分解する。
5. 改訂 D1 が `Top-1 +3 pt + CI下限>0`、contrast 80%、regression 5%、frequency差5 points以内、handmade 14/14を同時に通ってから D2 を実施する。
6. S200 は tokenizer/packing/checkpoint の **quality判断に使わない smoke**だけなら独立に実施可能。D1/D2/S200 の品質比較と3 seed runは上記 gate 通過後に行う。
7. quality winner 決定後にだけ N100 上で candidate K=5 whole-batch、tokenize込みの変換全体 p50/p95、RSS、model bytesを測定する。今回の GPU 4.332 ms/candidate は production Go の代用にならない。
