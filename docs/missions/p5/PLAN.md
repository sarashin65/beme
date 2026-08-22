# Issue #5: IME 専用小型 LM 学習プラン

作成日: 2026-08-22  
対象: BeME `main`（#6 Exact K-best / 文全体ラティス化後）  
状態: 構想・着手判断用。実装は本書の範囲外。

## 0. 結論

### 推奨する着手案

1. **0.5B デコーダーをゼロから学習しない。** 最初は日本語の事前学習済み **30M 級エンコーダー型 Cross-Encoder** を、IME の候補集合に対する listwise ranking でファインチューニングする。
2. 「自然さ」は文章の無条件生成尤度ではなく、`かな読み + 前後文脈 + 候補集合` の下での条件付き選択確率として学習する。頻度バイアスは、同じ読みに対する候補を同一 group 内で比較し、読み単位の均等サンプリング、頻出語 hard negative、文脈を入れ替えた対照ペアで抑える。
3. 学習元の第一候補は [`hotchpotch/japanese-reranker-tiny-v2`](https://huggingface.co/hotchpotch/japanese-reranker-tiny-v2)。MIT、`ruri-v3-pt-30m` ベース、3 layer / hidden 256、ONNX INT8 成果物があり、CPU 配備の検証を最短化できる。130M の現行 `japanese-reranker-base-v2` は teacher / 上限比較に限定する。
4. #4 の Wikipedia 100 MiB を再利用する。ただし、現在の `data/corpus.txt` は記事 ID を失っているため、本評価用データは同じ固定ソースから記事 ID を保持して再生成し、記事単位で train/dev/test を分ける。#4 の頻度はモデルの正解ラベルにはせず、頻出 distractor の抽出とサンプリング層化に使う。
5. **GPU 契約はまだ不要。** CPU でデータ品質、oracle recall、既存 tiny モデルの N100 レイテンシを先に測る。最小学習は無料 Colab または 24 GB GPU の時間借りで行う。30M 級の signal が確認できるまで Colab Pro の月額契約も A100 も買わない。
6. 主評価は新設する Wikipedia 文脈対照 test。AJIMEE は長文 CER、Mozc は無文脈回帰、handmade は重要 14 ケースの sentinel として使う。候補に正解がない例はリランカ精度と混ぜず、`oracle recall@k` と条件付き ranking 精度を併記する。

### 着手判断

**Phase 0〜1（データ作成・CPU ベースライン）は着手してよい。GPU 月額契約は見送る。**  
最初の 30M pilot が、未見記事上で既存順位を統計的に上回り、頻度バイアス対照ペアを解き、かつ INT8 の N100 p95 が変換全体 50 ms 以下になる見込みを示した時だけ本格学習へ進む。

---

## 1. 判断の前提となる実測

### 1.1 repo 内の確定事実

| 観測 | 数値 / 結論 | 含意 |
|---|---|---|
| Qwen2.5-0.5B Q4_K_M logits | handmade Top-1 0.9286、150.891 ms/候補（Intel N100） | 0.5B でも目的関数は頻出語を選び、レイテンシも満たさない |
| 「歴史資料…こうしょう」 | Qwen は `交渉 > 考証` | 無条件に近い生成尤度は「ありふれた語」を優先する |
| 現行 130M Cross-Encoder | 48.668 ms/候補（Intel N100） | 候補を逐次採点する現状では変換全体が数十 ms に収まらない |
| deepdive | 正解経路は 175/200（87.5%）でラティス上に存在。モデル誤りが主因 | 辞書追加だけでなく ranking / cost model が必要 |
| P1 `matrix.def` そのまま | CER 0.341433 → 0.344480（悪化） | 解析用 LM をそのまま IME 目的へ移すのは不可 |
| P0 後 oracle recall@5 | 旧測定 28/200 = 14.0% | 5-best リランカの完全一致上限が低い |
| k 拡大 + 既存 Cross-Encoder（#6 前） | k=5〜64 で出力変更 0/200、CER 改善 0 | 候補を増やすだけでも、汎用リランカを置くだけでも足りない |
| #4 Wikipedia 100 MiB | 7,548,547 tokens、約101万行、21,829 unique tuples | 小規模な task FT データ源として再利用可能 |
| #4 頻度コスト（#6 前） | AJIMEE CER 0.290080 → 0.274551、一方 handmade Top-1 1.0 → 0.857 | 頻度は有効な信号だが、そのまま強めると同音異義語を壊す |

根拠: [spec](../../spec.md)、[deepdive](../../deepdive-2026-08-21.md)、[P1 report](../../p1-offline-experiment-2026-08-21.md)、[P2 k-exp report](../../p2-kexp-report.md)、[#4 report](../../issue4-phase1-report.md)、[oracle recall report](../../oracle-recall-after-p0-2026-08-21.md)。

### 1.2 本プラン作成時の再測定（2026-08-22）

#6 後の現行 worktree、既定 `mock`、`top_k=5` を実行した。

| 辞書 | AJIMEE CER / exact | Mozc | handmade Top-1 / NDCG / MRR |
|---|---:|---:|---:|
| `dictionary.tsv` | 0.290080 / 7.0% | 88/498 | 1.0 / 0.992972 / 1.0 |
| #4 `dictionary_learned.tsv`（N=1,000, α=0.20） | 0.277928 / 7.0% | 90/498 | 0.857143 / 0.953679 / 0.928571 |

現行 Exact K-best で AJIMEE 200件の候補を再生成した oracle recall は次の通り。

| k | exact gold を含む件数 | oracle recall |
|---:|---:|---:|
| 1 | 14 | 7.0% |
| 5 | 29 | **14.5%** |
| 10 | 39 | 19.5% |
| 20 | 47 | 23.5% |
| 50 | 59 | 29.5% |
| 64 | 62 | 31.0% |

旧 14.0% から @5 が 1 件変わったため、以後は **14.5% を現行の観測値**とする。ただし 14.5% は AJIMEE の全文完全一致 @5 であり、ラティス上の経路存在率 87.5% や理論上限ではない。

`docs/p2-kexp-report.md` と `docs/issue4-phase1-report.md` の主測定は #6 前である。方向性の根拠には使えるが、今後のモデル比較は必ず現行候補集合で再測定する。

---

## 2. 学習方式と目的関数

### 2.1 三方式の比較

| 方式 | 利点 | 問題 | 判断 |
|---|---|---|---|
| ゼロから事前学習 | tokenizer、サイズ、目的を完全に制御できる | #4 の 755万 token は 0.5B に対して桁違いに少ない。語彙・文法知識を再取得する費用が大きい。IME ranking の正解信号も弱まる | **不採用**。30M 未満の研究用モデルでも、task FT が失敗した後の比較以外では行わない |
| 継続事前学習 | 日本語・IME ドメインへ寄せられる | Wikipedia の raw 頻度を再学習し、今回避けたい頻度 prior を強めうる。直接 ranking を学習しない | **第二候補**。direct FT がデータ不足で頭打ちの場合のみ、30M encoder に kana corruption / MLM を短く実施 |
| supervised fine-tuning | `かな + 文脈 + 候補` と正解順位を直接最適化できる。必要データ・GPU が最小 | 自動かな化の誤り、候補生成由来の偏り、overfit がある | **推奨**。事前学習済み日本語 tiny Cross-Encoder を listwise FT |

0.5B をゼロから学習するための corpus 規模は本 PoC の範囲を超える。言語モデルの compute-optimal 学習ではモデル規模と token 数を共に増やす必要があるという一次研究もあり、755万 token から 4.9億 parameter を作る案は支持されない（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）。

### 2.2 「自然さ」を条件付き ranking として定義する

1 学習例を以下とする。

- 読み: $r$
- 前文脈 / 後文脈: $L, R$
- BeME が生成した候補集合: $C=\{c_1,\dots,c_k\}$
- 元文から得た正解候補: $c^+$
- Cross-Encoder score（大きいほど良い）: $s_\theta(L,r,R,c_i)$

主目的関数は group 単位の listwise softmax cross-entropy とする。

$$
P_\theta(c_i\mid L,r,R,C)
=\frac{\exp(s_\theta(L,r,R,c_i)/T)}
       {\sum_{c_j\in C}\exp(s_\theta(L,r,R,c_j)/T)},
\qquad
\mathcal L_{\mathrm{list}}=-\log P_\theta(c^+\mid L,r,R,C)
$$

入力例:

```text
query:     [PRE] 歴史資料の内容を調べる [READING] こうしょう [POST] を行った
candidate: 考証
```

実際には候補を埋め戻した全文も入力し、左右文脈と候補の局所連続性を encoder が同時に読める形式を比較する。候補全件を 1 batch で採点する。

Viterbi cost は model 入力へ混ぜず、dev 上でのみ別スカラー $\lambda$ を校正する。

$$
S_i=s_\theta(L,r,R,c_i)-\lambda\,\widetilde{C}_{\mathrm{Viterbi}}(c_i)
$$

これにより「LM 単体の寄与」「既存 cost の寄与」「blend の寄与」を分離して測れる。

### 2.3 頻度バイアスをデータ / loss で防ぐ

- **読み group をサンプリング単位にする。** Wikipedia の出現回数に比例して sentence を流し込まず、同じ reading の上限数を設ける。
- **頻出表記を hard negative にする。** #4 の頻度表から、「交渉」のように corpus 頻度は高いが当該文脈では誤りの候補を優先投入する。
- **対照ペアを作る。** 同じ reading について、文脈 A では候補 X、文脈 B では候補 Y が正解になる pair を同じ split に置く。片方だけを学習させない。
- **候補集合内で正規化する。** 文全体の全 token を予測する causal LM loss ではなく、同音候補間の相対 score だけを直接最適化する。
- **正例/負例の候補頻度を記録する。** 高頻度候補だけで正解率が落ちていないか層別評価する。
- **context dropout は dev で選ぶ。** `[推測]` 候補値 `{0, 0.1, 0.2}`。文脈が空の時に壊れないことは学習するが、無文脈の頻度 prior を主目的にしない。
- **任意の補助 loss**: 同一 reading の高頻度誤候補に対する pairwise margin。主 loss を置き換えず、ablation で効果がある場合だけ残す。

---

## 3. モデルサイズとアーキテクチャ

### 3.1 推奨順

| 優先 | モデル | 用途 | 理由 |
|---:|---|---|---|
| 1 | `japanese-reranker-tiny-v2`（30M family、3×256） | student / 最初の FT | MIT、日本語、ranking 済み、ONNX INT8、CPU 配備向け |
| 2 | 同 `xsmall-v2`（10×256） | tiny が capacity-limited の時だけ | 同じ family で深さの scaling を比較できる |
| 3 | 現行 `japanese-reranker-base-v2`（130M、19×512） | teacher / quality ceiling | repo で動作実績あり。ただし N100 48.668 ms/候補 |
| 4 | Qwen2.5-0.5B | offline teacher / 対照 | Apache-2.0、0.49B。ただし現行 causal likelihood は頻度誤りと遅延が実測済み |

`japanese-reranker-tiny-v2` のモデルカードは、tiny が 15万 pair の CPU 推論で base より大幅に速いこと、ONNX の AVX2/ARM64 INT8 を提供することを示す。ただし測定 CPU は M4 Max であり、**N100 の速度は未測定**。Phase 0 の必須 gate とする。

### 3.2 0.5B で足りるか

- 容量として「足りない」という証拠はない。既存 0.5B は文脈の一部を扱える一方、失敗は `交渉 > 考証` という目的関数 / prior の問題として観測された。
- `[推測]` IME の有限候補 ranking に 0.5B は過大で、最終 CPU 配備の制約が先に律速する。0.5B Q4 は 397 MB、N100 で 150.891 ms/候補だった。
- よって問うべき順序は「0.5B で足りるか」ではなく、**30M の条件付き ranking で学習 signal が出るか → xsmall / 130M で scaling gain があるか**である。

### 3.3 encoder を選ぶ理由

- 前文脈だけでなく後方文字列も同時に読むため、bidirectional encoder が自然。
- 出力は有限候補の scalar score であり、生成能力や長い decode は不要。
- 候補を batch 化でき、decoder の候補ごとの autoregressive token 評価を避けられる。
- causal decoder は右文脈を含む全文尤度を採ると、候補以外の token loss が score を支配しやすい。

代替案:

- **bi-encoder**: context 側を cache できるが、短い同音候補との細かな相互作用が弱い。tiny Cross-Encoder が latency gate を落とした時の fallback。
- **character n-gram / linear ranker**: CPU-only の下限 baseline。モデル本命ではないが、データに signal があるかを GPU 前に確認できる。
- **seq2seq / 直接生成**: candidate ceiling を越えられる可能性はあるが、ranking 中心の設計と latency を崩す。oracle recall を改善しても不足することが確認されるまで別 issue にしない。

### 3.4 配備形

最終成果物は PyTorch checkpoint ではなく、batch 入力可能な **ONNX INT8** を第一候補とする。測定条件は Intel N100、候補 k=5、前後文脈込み、tokenize + inference + score blend を含む。モデル load は cold と warm を分ける。

採用基準:

- hard gate: warm p95 **50 ms/変換全体以下**（5候補合計。ms/候補ではない）
- stretch: warm p95 20 ms 以下
- `[推測]` model file 50 MB 以下、追加 RSS 100 MB 以下を設計目標とする

---

## 4. 学習データ

### 4.1 データ形式

Wikipedia 記事の一文から変換対象 span を切り出し、表層をかな読みに置換する。

```json
{
  "source": "wikipedia-20231101.ja",
  "article_id": "...",
  "left_context": "歴史資料の内容を調べるため、",
  "reading": "こうしょう",
  "right_context": "を行った。",
  "positive": "考証",
  "candidates": ["交渉", "考証", "校章", "高尚"],
  "viterbi_costs": [2.1, 2.4, 2.5, 2.7],
  "frequency_bucket": {"positive": "mid", "hard_negative": "high"}
}
```

作成規則:

1. fugashi/IPADIC 等で surface → reading を得る。
2. 変換対象は単語 1 個だけでなく、分割誤りを含む短い span も採る。
3. 現行 BeME Exact K-best で候補を生成する。
4. 元 surface が候補内にある group を ranking 学習へ使用し、候補外の group は retrieval 失敗として別集計する。
5. negative は現行 top 候補、同音表記、分割違い、カタカナ漏れ、高頻度 distractor を混ぜる。
6. article ID 単位で split し、同一文・同一近接文を跨がせない。
7. AJIMEE の `original_text`、handmade、Mozc の文字列と exact / normalized / near-duplicate が一致する記事を train から除く。AJIMEE は Wikipedia 由来のため、これをしない評価は leakage になる。
8. 自動かな化した label をランダム 200 group 以上目視し、採用基準を正解率 98%以上とする。固有名詞、多音語、数値読みは別 bucket にし、誤りが多ければ pilot から外す。

### 4.2 量

| 段階 | group 数 | 目的 |
|---|---:|---|
| smoke | `[推測]` train 10,000 / dev 1,000 / test 2,000 | 目的関数に signal があるか。最小 GPU 実験 |
| pilot | `[推測]` train 50,000 / dev 5,000 / test 10,000 | loss / hard-negative / context ablation |
| scale | `[推測]` 100,000〜1,000,000 train | scaling curve で data-limited と判定された場合のみ |

件数は予算見積りであって実測済みの最適値ではない。最初に 100 MiB から得られる「gold が候補内にある有効 group 数」と reading 分布を測り、決め直す。全量 Wikipedia / CC-100 を先に取得しない。

### 4.3 #4 の成果の使い方

再利用するもの:

- 固定取得元 `wikimedia/wikipedia`, config `20231101.ja`
- 100 MiB の corpus 取得経路
- fugashi/IPADIC の解析経路
- 755万 token の頻度集計
- 高頻度語と `(reading, surface, pos)` 集計

そのまま使わないもの:

- `dictionary_learned.tsv` を LM 学習 baseline へ混ぜること。現行再測定でも AJIMEE は改善したが handmade 2/14 を壊したため、独立 ablation とする。
- raw 行数を train/test split に使うこと。`data/corpus.txt` は記事境界情報を十分保持しない。
- 出現頻度をそのまま正解重みとすること。これは #2 の bias を再注入する。

### 4.4 ライセンス / provenance

| データ | 用途 | 一次情報 | 判断 |
|---|---|---|---|
| `wikimedia/wikipedia` `20231101.ja` | train/dev/test 候補 | [dataset card](https://huggingface.co/datasets/wikimedia/wikipedia) は CC BY-SA 3.0 / GFDL と表示 | 使用可。ただし dump/config/revision、記事 ID、取得日、帰属、license を manifest に残す |
| Wikimedia 現行 text | 将来 dump | [Wikimedia Terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content) | 現行 contribution 条項は CC BY-SA 4.0。実際に使う dump の表示を優先する |
| CC-100 ja（約15 GB） | 将来候補 | [dataset card](https://huggingface.co/datasets/statmt/cc100) | **license: unknown**。初期学習には使わない。URL 単位の権利・PII・opt-out を整理できる時だけ再検討 |
| AJIMEE JWTD v2 | evaluation only | repo `data/README.md` | CC BY-SA 3.0。train へ混ぜない |
| Mozc `oss.tsv` | regression only | repo `data/README.md` | BSD 3-Clause。train へ混ぜない |
| handmade 14 | sentinel | repo 内自作 | train へ混ぜない |

#4 文書の「Wikipedia は CC BY-SA 4.0」という一般記述と、実際に使用した Hugging Face dataset card の `CC BY-SA 3.0 / GFDL` は一致しない。本タスクでは **実際の pinned artifact の dataset card を manifest に記録**する。

学習済み weights が share-alike の対象になるかは本書で法的断定をしない。公開前に model card へ data provenance と帰属を記載し、weights の license は別途レビューする。

---

## 5. GPU 契約の判断

### 5.1 必要スペック

| 実験 | 推奨 VRAM | GPU 要否 | 備考 |
|---|---:|---|---|
| データ生成、oracle、linear baseline、N100 latency | 0 | 不要 | CPU で先に完了させる |
| 30M tiny full FT、seq≤128 | `[推測]` 8 GB 最低 / 16 GB 余裕 | 任意だが GPU 推奨 | 無料 T4 級で pilot 可能 |
| 130M full FT / teacher scoring | `[推測]` 16〜24 GB | 推奨 | mixed precision、batch 調整 |
| 0.5B LoRA | `[推測]` 24 GB | 必要 | 今回は比較用に限定 |
| 0.5B full FT / continued pretrain | `[推測]` 40〜80 GB | 必要 | 費用対効果がなく当面実施しない |
| 0.5B scratch | 単 GPU判断の範囲外 | 必要 | corpus・compute とも不採用 |

VRAM と所要時間は実測前の保守的見積り。最初の 500〜1,000 step で peak VRAM と step/s を記録し、残 step と比較本数から費用を再計算する。

### 5.2 選択肢と費用感（2026-08-22 時点）

#### Colab

- Free で最初の tiny pilot を試す価値がある。
- [Colab FAQ](https://research.google.com/colaboratory/faq.html) は、GPU 種別・usage limit・availability が変動し、paid plan でも特定 resource は保証されないとしている。通常 runtime は最大12時間、Pro+ は compute unit があれば最大24時間。
- 日本アカウントの表示価格は sign-in 後の [pricing page](https://colab.research.google.com/signup) で確認が必要で、本書では未確認の固定価格を引用しない。
- 再現実験を連続実行する用途では preemption と GPU 不定が弱点。月額契約は convenience への支払いであり、この規模の必須条件ではない。

#### 時間借り GPU

[Runpod 公式価格](https://www.runpod.io/pricing):

- RTX 4090 24 GB: Community **$0.34/h**、Secure **$0.74/h**
- RTX 3090 24 GB: Community **$0.22/h**、Secure **$0.50/h**
- A40 48 GB: Community **$0.35/h**、Secure **$0.44/h**
- A100 PCIe 80 GB: Community **$1.19/h**、Secure **$1.39/h**

`[推測]` 30M pilot が 1〜3 GPU-hour なら 4090 compute は $0.34〜$2.22、1 config が 12時間でも $4.08〜$8.88。storage、停止忘れ、税は別。

[Lambda 公式価格](https://lambda.ai/instances):

- V100 16 GB: **$0.79/GPU-hour**
- A100 40 GB: **$1.99/GPU-hour**
- A100 80 GB: **$2.79/GPU-hour**

30M では A100 は不要。供給、region、永続 disk、データ転送、停止単位を含めた総額で選ぶ。

### 5.3 契約 gate

1. **今**: 契約しない。Phase 0〜1 は CPU、必要なら Colab Free。
2. **最小 FT**: 24 GB GPU を 1 回だけ時間借り。月額契約しない。
3. pilot の 1,000 step 実測から次を計算する。
   - 1 run の所要時間
   - 3 seed × ablation 数 × model size 数
   - checkpoint / storage 費用
4. **Colab Pro を選ぶ条件**: 2か月以上、毎月反復的に notebook GPU を使い、動的 GPU / 中断を許容し、account に表示された compute unit 単価が時間借り総額を下回る。
5. **時間借りを選ぶ条件**: GPU 型を固定したい、数日で experiment matrix を終える、実行時間を再現したい。今回はこちらが第一候補。
6. **A100 へ上げる条件**: 30M / 130M で signal があり、実測 peak VRAM または wall time が 24/48 GB GPU を明確に律速する場合だけ。

---

## 6. 検証方法

### 6.1 評価セットの役割

| suite | 測るもの | 期待値 / 注意 |
|---|---|---|
| Wikipedia 文脈付き（新設、Step 3） | 主指標。未見記事・同一 reading の文脈対照・頻度 distractor | ranking Top-1、MRR/NDCG、対照ペア両方正解、oracle recall。学習効果の採否を決める |
| AJIMEE 200 | 長い入力の end-to-end CER / exact、Wikipedia domain | 現行 baseline 0.290080 / 7%。@5 oracle 14.5%なので exact 改善の上限を併記。Wikipedia train との重複除去必須 |
| Mozc 498 | 無文脈 corner-case regression | context が空なら第2ランカーを skip し、88/498 を bit-for-bit 維持するのが基本期待 |
| handmade 14 | 「考証/交渉」等の重要 sentinel | Top-1 14/14、NDCG 0.992972、MRR 1.0 を維持。小さすぎるため主学習指標にしない |

AJIMEE には `context_text` がある項目もあるが、「同じ reading で文脈だけを変えて正解が入れ替わる」対照 dataset ではない。#6 前の既存 Cross-Encoder が候補を増やしても 0/200 出力を変えなかった事実は、Wikipedia Step 3 の対照評価を先に作る理由であり、AJIMEE を捨てる理由ではない。

### 6.2 ranking と retrieval を分離する指標

全 test で以下を同時に出す。

1. `oracle recall@k`: gold が候補集合に存在する割合
2. `conditional Top-1`: gold が存在する group のうち rank 1 にできた割合
3. `recoverable-error recovery`:
   $$
   \frac{\#(baseline\ wrong,\ gold\ in\ C,\ model\ correct)}
        {\#(baseline\ wrong,\ gold\ in\ C)}
   $$
4. `regression rate`: baseline correct → model wrong
5. MRR@k / NDCG@k
6. end-to-end CER / exact match
7. 頻度 bucket、reading ambiguity 数、span 長、左/右文脈長、固有名詞別の層別値
8. p50 / p95 latency、cold load、warm batch、RSS、model bytes

候補外 miss をリランカ誤りへ数えない一方、end-to-end 指標からは除外しない。この二面表示で「retrieve の天井」と「ranker の回収力」を混同しない。

### 6.3 比較プロトコル

1. dictionary SHA、model SHA、corpus revision、split manifest、seed を固定。
2. test は着手前に凍結し、hyperparameter / blend weight は dev だけで決める。
3. 同一 candidate set を保存し、次を同条件で比較する。
   - Viterbi only
   - #4 frequency cost（独立 ablation）
   - off-the-shelf tiny
   - fine-tuned tiny
   - fine-tuned tiny + Viterbi blend
   - xsmall / 130M は tiny に signal が出た場合だけ
4. model score、Viterbi cost、blend score、順位、gold rank を候補ごとに保存。
5. pilot は探索 1 seed でよいが、Go 判定は 3 seed。paired bootstrap 95% CI を出す。
6. N100 latency は同じ warm-up、thread 数、context 長、batch k で複数回測り、候補単位でなく変換単位の p95 を採る。
7. cascade を評価する場合は、呼び出し率と品質の curve を出し、最良点だけ報告しない。

### 6.4 Go / No-Go 基準

数値は事前に固定する採用基準であり、既に達成した実測値ではない。

**最小 pilot の Go:**

- Wikipedia held-out の conditional Top-1 が Viterbi と off-the-shelf tiny の双方より **+3 percentage points 以上**
- paired bootstrap 95% CI の改善下限が 0 より大きい
- recoverable-error recovery 20%以上、baseline-correct regression 5%以下
- 文脈対照 subset の「両文脈正解」80%以上
- 高頻度 distractor subset が全体より 5 points 超悪化しない

**統合の Go:**

- handmade 14/14 と Mozc no-context 出力を維持
- AJIMEE CER が fresh baseline を悪化させない。改善する場合は paired CI と変更例を報告
- Wikipedia 主 test の改善が 3 seed で再現
- ONNX INT8、N100、k=5 batch の warm p95 が 50 ms/変換以下
- dataset / model license、attribution、再生成 manifest が揃う

いずれかを落としたら 0.5B へ即拡大しない。失敗が data、candidate recall、objective、capacity、runtime のどこかを ablation で分類してから次を決める。

---

## 7. フェーズとマイルストーン

### M0: 現状固定（GPU なし）

本書作成時に baseline / #4 learned / oracle recall を #6 後で再測定済み。実タスク開始時には同じ結果を machine-readable artifact として保存し、dictionary / code revision を付ける。

完了条件:

- baseline、oracle recall@1/5/10/20/50/64、候補 latency の固定
- #4 frequency cost を混ぜない baseline の確定
- off-the-shelf tiny の N100 PyTorch / ONNX FP32 / INT8 batch benchmark

### M1: CPU データ gate（GPU なし）

- pinned Wikipedia を記事 ID 付きで取得
- AJIMEE / handmade / Mozc 重複を除去
- article split を固定
- 10k / 1k / 2k group を生成
- gold-in-candidate 率、reading 分布、hard-negative 分布を確認
- 200 group 目視 audit
- linear ranker と off-the-shelf tiny で data signal を確認

**No-Go:** label 正解率 98%未満、gold-in-candidate が低く有効 10k group を作れない、train/test の reading / article leakage が残る。

### M2: 最小学習実験（最初の GPU、1 run）

これが「小さく検証する第一歩」。

- model: `japanese-reranker-tiny-v2`
- data: train 10k / dev 1k / test 2k、記事 disjoint
- negatives: 現行 top + 高頻度 hard negative、1 group `[推測]` 3〜5 negatives
- objective: listwise CE のみ
- context: `[推測]` max 128 subword
- train: `[推測]` 1 epoch、探索 1 seed
- GPU: Colab Free または 24 GB 時間借り。月額契約なし
- compare: Viterbi、off-the-shelf tiny、FT tiny
- export: ONNX INT8、N100 benchmark

この実験の問いは一つだけ: **「balanced conditional ranking を学ぶと、未見文脈で頻度 prior より文脈を選べるか」**。

### M3: pilot 拡大

M2 が Go の時だけ:

- 50k〜100k group
- 3 seed
- hard-negative、context dropout、pairwise 補助 loss の ablation
- tiny → xsmall の scaling 比較
- 130M teacher から distillation を比較
- Viterbi blend / cascade を dev 校正

ここで初めて継続的 GPU 予算を見積もる。1,000 step の実測から総 GPU-hour を計算し、時間借りと Colab の account 表示価格を比較する。

### M4: data scaling / optional continued pretraining

M3 が data-limited と示した場合だけ:

- 100 MiB → Wikipedia 全量の scaling curve
- 30M encoder の kana corruption / MLM continued pretraining
- CC-100 は license / provenance gate を通るまで不使用
- 0.5B は offline teacher の gain が 30M/130M を明確に上回る時だけ比較

### M5: ローカル配備 gate

- ONNX INT8 artifact
- batch scoring
- N100 p50/p95、RSS、cold/warm
- AJIMEE / Mozc / handmade / Wikipedia locked test
- model card、data manifest、attribution

全 Go 基準を満たした時だけ BeME runtime 統合を別 issue で行う。

---

## 8. リスクと「やらなくていいこと」

### 8.1 主リスク

| リスク | 検出 | 対策 |
|---|---|---|
| gold が候補にない | oracle recall@k | ranker と retrieval を分離。候補外なら LM サイズを上げない |
| Wikipedia / AJIMEE leakage | article ID、fingerprint、near-duplicate | article split、評価原文の除外、manifest |
| 自動かな化の誤 label | 目視 audit、固有名詞 bucket | round-trip filter、曖昧読み除外、98% gate |
| 頻度 bias 再発 | 頻度 bucket、対照 pair | reading 均等 sampling、頻出 hard negative、listwise loss |
| synthetic candidate bias | candidate source 別評価 | 現行 top、深い lattice、辞書同音、分割負例を混合 |
| tiny の capacity 不足 | tiny/xsmall/130M scaling | gain が単調な時だけ拡大 |
| CPU latency 超過 | N100 whole-batch p95 | short context、batch、INT8、cascade、必要なら bi-encoder |
| #4 cost と LM の効果混同 | 同一候補 set の ablation | frequency dictionary は独立条件に保つ |
| license 不明 corpus | manifest / dataset card | CC-100 を初期対象外にする |
| test 過適合 | locked test、3 seed、CI | dev 以外で weight を調整しない |

### 8.2 実測により再試行不要な方向

以下は新しい仮説・新しい評価条件なしに繰り返さない。

1. **IPADIC word cost + `matrix.def` へそのまま切替** — CER 0.341433 → 0.344480。
2. **16 POS 遷移表の小幅 tuning** — 最良でも CER -0.002、matrix 分散の79%を失う。
3. **beam width 64→256 だけの拡大** — deepdive で出力・CER 変化なし。
4. **#6 前の構成で k=5→64 + 汎用 Cross-Encoder** — 候補は約10倍、出力変更 0/200、CER 改善 0。
5. **AJIMEE 上でリランカ weight だけ調整** — 候補外問題と score 問題を分離できず、同じ200件への過適合になる。
6. **Qwen2.5-0.5B の平均 token 尤度をそのまま「自然さ」とする** — `交渉 > 考証` と N100 150.891 ms/候補を再現済み。
7. **Wikipedia raw 頻度だけを強める** — #4 は CER を改善した一方 handmade Top-1 を 1.0→0.857に悪化させた。
8. **0.5B scratch を先に始める** — data / objective / latency の三 gate を一つも解決しない。
9. **CC-100 を「公開だから自由」とみなして投入** — dataset card の license は unknown。
10. **GPU 契約を先に買う** — M0〜M1 と最小 30M pilot は契約なしで判断可能。

### 8.3 別 issue にすべき方向

- seq2seq による候補直接生成
- lattice / 辞書の gold coverage 改善
- user personalization / 履歴学習
- CC-100 の権利・PII・opt-out 調査
- runtime integration と IME UI の非同期推論

これらは本タスクの「文脈付き候補 ranking を小型モデルで学べるか」という問いに混ぜない。

---

## 9. 最終チェックリスト

### CPU 着手

- [ ] 現行 revision の baseline / oracle artifact を保存
- [ ] tiny ONNX INT8 を Intel N100 で whole-batch 測定
- [ ] article ID 付き Wikipedia split manifest を作成
- [ ] 評価 corpus との重複を除去
- [ ] 10k/1k/2k group と 200件 audit を完了

### 最初の GPU 実験

- [ ] 30M tiny、listwise CE、1 epoch の 1 run
- [ ] Viterbi / off-the-shelf / FT の同一候補比較
- [ ] 頻度 bucket / 文脈対照 / regression を出力
- [ ] ONNX INT8 の N100 p95 を測定
- [ ] Go 基準を満たすまで月額契約しない

### 本格着手

- [ ] 3 seed で改善再現
- [ ] GPU-hour を実測から算出
- [ ] 時間借りと Colab account 表示価格を比較
- [ ] license / attribution / model card 方針を確定
- [ ] 全 suite と latency gate を通過

---

## 10. 参照一次情報

### repo / issue

- [Issue #5](https://github.com/sarashin65/beme/issues/5)
- [Issue #2](https://github.com/sarashin65/beme/issues/2)
- [Issue #3](https://github.com/sarashin65/beme/issues/3)
- [Issue #4](https://github.com/sarashin65/beme/issues/4)
- [PR #6: Exact K-best / 文全体ラティス化](https://github.com/sarashin65/beme/pull/6)
- [BeME spec](../../spec.md)
- [deepdive](../../deepdive-2026-08-21.md)
- [P1 offline experiment](../../p1-offline-experiment-2026-08-21.md)
- [P2 k expansion report](../../p2-kexp-report.md)
- [#4 Phase 1 report](../../issue4-phase1-report.md)
- [P0 後 oracle recall](../../oracle-recall-after-p0-2026-08-21.md)

### model / data / compute

- [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B) — Apache-2.0、0.49B、24 layers
- [japanese-reranker-tiny-v2 model card](https://huggingface.co/hotchpotch/japanese-reranker-tiny-v2) — MIT、tiny architecture、ONNX INT8
- [japanese-reranker-base-v2 model card](https://huggingface.co/hotchpotch/japanese-reranker-base-v2) — MIT、130M base family、size/quality/latency comparison
- [Wikimedia Wikipedia dataset card](https://huggingface.co/datasets/wikimedia/wikipedia) — 使用 dataset の license / article schema
- [Wikimedia Terms of Use §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content) — text licensing / attribution
- [CC-100 dataset card](https://huggingface.co/datasets/statmt/cc100) — ja 約15 GB、license unknown
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — scratch pretraining の data/compute 判断
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) / [pricing](https://colab.research.google.com/signup) — dynamic limits、runtime、account-local pricing
- [Runpod pricing](https://www.runpod.io/pricing) — GPU-hour 価格
- [Lambda GPU instances](https://lambda.ai/instances) — GPU-hour 価格
