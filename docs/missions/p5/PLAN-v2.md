# Issue #5: IME 専用小型 decoder LM 学習プラン v2

作成日: 2026-08-22  
対象: BeME `main`（#6 Exact K-best / 文全体ラティス化後）  
状態: 構想・着手判断用。実装は本書の範囲外。  
旧版: [PLAN.md](./PLAN.md)（30M encoder Cross-Encoder を本命とした判断経緯を参照用に保存）

## 0. 結論

1. **本命を decoder-only causal LM に変更する。** IME 用の小型言語モデルを実際に学習し、有限候補を条件付き尤度で選ぶ。30M encoder Cross-Encoder は削除せず、decoder の速度問題を切り分ける副次比較に下げる。
2. 採用は好みではなく、同じ BeME task data と locked test を使った **約 200M のゼロから学習**対**既存 0.5B の FT / 継続事前学習**の精度で決める。初期値、parameter 数、tokenizer、総学習データが異なるため完全な同条件にはならない。共通条件と不可避な差を分けて報告する。
3. 最初の GPU run は自前 RTX 3060 Ti 8 GB 上の **Qwen2.5-0.5B base QLoRA task FT、10k group、1 seed**とする。0.5B への関心を最短で検証し、データ、loss、評価、VRAM、速度のパイプラインを一度に確認できるためである。
4. 3060 Ti では、0.5B QLoRA / LoRA と約 150–250M scratch を試す。0.5B full-parameter scratch は、標準 AdamW の parameter 関連状態だけで約 7.8 GB（10進）となり、activation と CUDA workspace を加えると 8 GB を超えるため行わない。
5. #4 の Wikipedia 100 MiB / 7,548,547 fugashi token は、データ経路と smoke test には使えるが、約 200M scratch の性能判定には足りない。Hoffmann et al. の compute-optimal 表の約 20 token/parameter を尺度にすると、200M には約 4B LM token、0.5B には約 10B LM token が目安になる。これは一般 LM の尺度であって IME 専用 LM の必要量を保証しないが、100 MiB を本学習 corpus とみなす根拠はない。
6. 有料 Colab の 0.5B scratch は自動昇格先ではない。ローカル実測で「data を増やすほど scratch が伸びる」「目的指標で decoder に signal がある」「割当 GPU / runtime で完走可能」の三条件を満たした時だけ進む。Colab は paid でも GPU 型・上限・availability が保証されないため、接続後の実機測定を gate にする。
7. 研究上の勝者は精度で決める。production 採用にはその後、Intel N100 上の変換全体 latency と memory の gate を別に通す。精度が同等なら「同等」と報告し、scratch 勝利とはしない。

---

## 1. 前提、確定事実、見積もり

### 1.1 repo 内の確定事実

旧版で #6 後に再測定済みの値を比較の固定点とする。

| 観測 | 値 | 本計画への含意 |
|---|---:|---|
| Qwen2.5-0.5B Q4_K_M | handmade Top-1 0.9286、Intel N100 150.891 ms/候補 | off-the-shelf の無条件に近い尤度は精度・速度とも未達 |
| 「歴史資料…こうしょう」 | `交渉 > 考証` | raw 頻度 prior ではなく文脈条件付き task adaptation が必要 |
| #6 後 AJIMEE baseline | CER 0.290080、exact 7.0% | end-to-end 比較の基準 |
| #6 後 AJIMEE oracle | recall@5 14.5%、@64 31.0% | candidate retrieval と LM ranking を分離して測る必要がある |
| deepdive | 正解経路 175/200（87.5%） | lattice 上限と K-best 上限も分ける |
| #4 Wikipedia 100 MiB | 7,548,547 fugashi token、約101万行、21,829 unique tuples | smoke / tokenizer / task group 生成経路として再利用可能 |
| #4 頻度 cost | AJIMEE 改善、handmade Top-1 1.0→0.857 | 頻度は有効だが単独強化は危険 |

根拠: [spec](../../spec.md)、[deepdive](../../deepdive-2026-08-21.md)、[P1 report](../../p1-offline-experiment-2026-08-21.md)、[P2 k-exp report](../../p2-kexp-report.md)、[#4 report](../../issue4-phase1-report.md)、[oracle recall report](../../oracle-recall-after-p0-2026-08-21.md)、旧版 [PLAN.md](./PLAN.md)。

### 1.2 hardware / software の確定事実

- RTX 3060 Ti は Ampere、VRAM 8 GB。NVIDIA の Ampere guide は FP16 と BF16 Tensor Core を記載している。実際に使える dtype は PyTorch / CUDA build を含めて実機 probe する。
- NVIDIA は WSL2 で Linux application と CUDA を使えるとしている。Windows 側 NVIDIA driver が必要で、**WSL 内へ Linux display driver を入れてはならない**。
- Qwen2.5-0.5B base の model card は causal LM、0.49B parameter、24 layers、Apache-2.0、日本語を含む 29+ languages と記載している。
- QLoRA は凍結した 4-bit base weights を通して LoRA adapter だけを学習する。bitsandbytes は NF4/FP4 と 8-bit optimizer を CUDA compute capability 6.0+ で支援し、現行 wheel は Linux / Windows x86-64 の `sm86` を含む。
- PyTorch activation checkpointing は forward 中間 tensor を保持せず backward で再計算し、compute と memory を交換する。
- Colab は GPU 型、usage limit、maximum lifetime を公開固定しておらず、paid plan でも availability は変動する。通常 runtime は最大12時間、Pro+ は compute unit があれば最長24時間と FAQ にある。

### 1.3 表記

- **確定事実**: repo の保存済み実測、または一次情報に明記された内容。
- **`[見積]`**: parameter 数、dtype、optimizer state からの算術、または未実測 config の予算値。
- **`[推測]`**: 期待する精度、適切な hyperparameter、必要な domain data 量など、実験前に確定できないもの。

---

## 2. 学習対象と decoder の使い方

### 2.1 本命: IME 専用 causal decoder LM

decoder は二段階で学習する。

1. **causal LM 学習**
   - scratch arm: train split の raw Japanese text から random initialization で事前学習する。
   - FT arm: 既存 0.5B を初期値にし、同じ raw train documents で継続事前学習する条件と、継続事前学習なしの条件を分ける。
2. **IME task adaptation**
   - 全 arm に同一の `left context + reading + optional right context → gold surface` を与え、answer 部分だけ causal LM loss を取る。
   - 推論時は BeME 候補 $c_i$ ごとに answer token と EOS の条件付き対数尤度を計算し、候補集合内で順位付けする。

入力例:

```text
<left>歴史資料の内容を調べるため、</left>
<reading>こうしょう</reading>
<right>を行った。</right>
<answer>考証</answer>
```

score は次を基本とする。

$$
s_\theta(c_i)=\sum_{t=1}^{|c_i|+1}
\log P_\theta(y_t\mid L,r,R,y_{<t})
$$

- EOS を含め、候補文字列の途中だけ高い出力を防ぐ。
- token 長による bias が出るため、sum、token 平均、文字長 penalty の候補を **dev だけ**で選び、test では固定する。
- right context が runtime で取得できる場合とできない場合を混ぜない。`left-only` と `left+right` を別条件で報告する。
- raw 文全体 perplexity は補助指標とし、採用指標にしない。既存実測で raw likelihood が `交渉 > 考証` を起こしたためである。
- Viterbi cost との blend は decoder 単体の比較後、dev だけで係数を校正する。

### 2.2 比較する decoder arm

| ID | model / initialization | raw LM stage | task stage | 3060 Ti |
|---|---|---|---|---|
| D0 | Qwen2.5-0.5B base frozen | なし | なし | inference baseline |
| D1 | Qwen2.5-0.5B base | なし | QLoRA SFT | **最初の 1 run** |
| D2 | Qwen2.5-0.5B base | 同一 raw train docs で QLoRA continued pretraining | D1 と同一 task SFT | 可。D1 に signal が出た後 |
| S200 | custom decoder `[見積]` 180–220M、random init | scratch causal pretraining | D1 と同一 task SFT | 可。data 拡張後 |
| S500 | Qwen2.5-0.5B config / tokenizer、random init | scratch causal pretraining | D1 と同一 task SFT | **Colab gate 後のみ** |

S200 の初期候補は `[見積]` 18 layers、hidden 896、14 attention heads、FFN 2432、32k tied vocabulary、non-GQA で約204M parameter とする。実行前に model を instantiate して `sum(p.numel())` を保存し、150–250M を外れたら config を直す。architecture 値は最適と実証されたものではない。

S500 は Qwen と同じ architecture / tokenizer を random initialization して D1/D2 と比べる。これにより 0.5B での「pretrained initialization の有無」を分離する。custom tokenizer まで含めた 0.5B scratch は別 ablation であり、最初の Colab run に混ぜない。

### 2.3 副次案: 30M encoder Cross-Encoder

旧版の `hotchpotch/japanese-reranker-tiny-v2` listwise FT は次の位置付けで残す。

- decoder が精度 signal を示すが N100 latency gate を落とした時の低遅延対照。
- task data 自体に ranking signal があるかを安く確かめる sanity check。
- decoder と encoder の quality / latency frontier を測る副次実験。

**encoder の勝敗で decoder scratch 対 FT の比較を打ち切らない。** 一方、production 統合では decoder が latency を満たさず encoder が同等精度なら encoder を選べる。研究本命と配備判断を混同しない。

---

## 3. ゼロから学習対 FT の比較設計

### 3.1 比較で固定するもの

- raw corpus の source revision、article / document split、除外 fingerprint。
- task group の train/dev/test、候補集合、gold、Viterbi cost。
- S200 / D1 / D2 が見る IME task train examples と epoch 数。
- locked Wikipedia context test、AJIMEE、Mozc、handmade。
- evaluation code、候補 score の保存形式、random seed list。
- dev で選ぶ項目と範囲。test を見て learning rate、length penalty、blend weight を変えない。
- raw LM data exposure は同じ **document set / UTF-8 bytes**を基本単位にする。tokenizer が異なるため、各 arm の native token 数も併記する。

### 3.2 固定できない差と解釈

| 差 | 理由 | 報告方法 |
|---|---|---|
| 事前知識 | D1/D2 は Qwen の非公開規模の prior training を継承する | 実用上の scratch 対 FT 比較では意図した差。公平化したと主張しない |
| tokenizer | S200 の32k custom tokenizer と Qwen tokenizer は異なる | perplexity を直接比較せず、bytes-per-byte / bits-per-byte と task accuracy を使う |
| parameter 数 | S200 約200M、D1/D2 0.49B | parameter、trainable parameter、FLOPs、wall time を併記 |
| optimizer | S200 は full-parameter AdamW、D1/D2 は QLoRA | 8 GB で実行可能な実用 recipe 同士の比較と明記 |
| raw LM token 数 | tokenizer で count が変わる | 同一 raw docs に加え、両 tokenizer の token 数を保存 |

この比較が答えるのは「手元の data / compute で、どちらが BeME 精度を出すか」である。初期化だけの科学的比較は、同じ 0.5B config / tokenizer を使う S500 対 D2 で行う。

### 3.3 実験 matrix

1. **baseline**: Viterbi only、#4 frequency cost、D0。
2. **first signal**: D1、1 seed。
3. **local comparison**: D1 対 D2 対 S200、同一 locked protocol、探索 1 seed。
4. **decision run**: 上位 recipe を 3 seed。各 seed の direction と paired bootstrap 95% CI を報告。
5. **optional scale**: gate を通った時だけ S500 対 D2。
6. **secondary**: 30M encoder CE。decoder の主比較表とは分ける。

### 3.4 採用規則

主指標は Wikipedia 文脈付き locked test の `conditional Top-1@k` とする。

- S200/S500 採用: best FT との差の paired bootstrap 95% CI 下限が 0 より大きく、3 seed 全て同方向。AJIMEE CER、handmade、Mozc の hard gate も通る。
- FT 採用: 逆方向に同じ条件を満たす。
- CI が 0 を跨ぐ: **精度同等 / 未決**。scratch が勝ったと解釈しない。追加 data point を取るか、production では latency、model bytes、学習費用を tie-break にする。
- raw LM loss が良くても IME 主指標が負ければ採用しない。

---

## 4. 学習データ

### 4.1 二種類の data artifact

1. **raw LM corpus**
   - `source`, revision, article ID, text fingerprint, license, split を持つ。
   - article 単位で train/dev/test を分ける。
   - tokenizer は raw train のみで学習し、dev/test は vocabulary 学習へ入れない。
2. **IME task group**

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

作成規則は旧版を引き継ぐ。

- fugashi/IPADIC 等で surface → reading を得る。
- 単語だけでなく短い span と分割違いを含める。
- 現行 BeME Exact K-best で候補を固定する。
- gold-in-candidate group は ranking、候補外は retrieval miss として別集計する。
- current top、同音表記、分割違い、カタカナ、高頻度 distractor を negative に混ぜる。
- reading ごとの上限を設け、Wikipedia 頻度比例だけで sampling しない。
- 同じ reading で文脈により正解が入れ替わる対照 pair を作る。
- AJIMEE / Mozc / handmade と exact、normalized、near-duplicate が一致する文書を train から除外する。
- ランダム200 group 以上を目視し、label 正解率98%以上を data gate とする。

### 4.2 #4 の 100 MiB で足りるか

**結論: smoke には足りるが、200M scratch の採否には足りない。**

- #4 の 7,548,547 は fugashi token であり、S200 / Qwen tokenizer の LM token と同じではない。まず両 tokenizer で再計数する。
- `[見積]` 7.55M / 200M = 0.038 observed token/parameter。Hoffmann et al. の Table 3 は 400M parameter に 8B token、すなわち約20 token/parameter を compute-optimal 例として示す。線形尺度なら200Mで約4B、0.5Bで約10B token になる。
- `[見積]` 200M の4Bに対し #4 は token 定義差を無視しても約1/530。0.5B の10Bに対して約1/1,325。
- 同じ100 MiBを何百 epoch も反復しても unique information は増えない。train loss だけ下がる場合は memorization と判定する。
- Chinchilla 則は一般 LM の compute-optimal frontier であり、IME domain の必要十分量ではない。4B / 10B を絶対下限とはしないが、data scaling curve なしに少量 scratch を不採用とも採用とも判定しない。

### 4.3 data 拡張順

| 段階 | data | 用途 | 判断可能なこと |
|---|---|---|---|
| A | #4 100 MiB | tokenizer / packing / loss / checkpoint smoke | 実装が動くか。精度優劣は判定不可 |
| B | pinned 日本語 Wikipedia の利用可能全量 | S200 の data scaling | held-out bpb と task accuracy の傾き |
| C | license / provenance を確認した追加日本語 corpus | 1B–4B token 方向の拡張 | S200 の本比較、S500 gate |

件数を先に推測で固定せず、各 source を native tokenizer token、UTF-8 bytes、document 数、重複除去率で inventory する。`[推測]` S500 の Colab exploratory run は少なくとも1B quality-controlled unique LM token を一つの暫定 floor とし、それ未満で進むなら local scaling curve が根拠を示す場合だけとする。10B は compute-optimal reference であり必達値ではない。

### 4.4 license / provenance

- `wikimedia/wikipedia` `20231101.ja` の dataset card は CC BY-SA 3.0 / GFDL と表示する。dump/config/revision、記事 ID、取得日、帰属、license を manifest に残す。
- Wikimedia 現行 Terms の contribution 条項と pinned artifact の表示を混ぜず、実際に使う artifact の表示を優先する。
- CC-100 の dataset card は license unknown。初期拡張へ入れない。
- AJIMEE は evaluation only、Mozc は regression only、handmade は sentinel とし、train へ混ぜない。
- 学習済み weights の license は本書で法的断定しない。公開前に provenance と attribution を別途レビューする。

---

## 5. 評価 protocol

### 5.1 評価 set の役割

| suite | 役割 | 指標 / hard gate |
|---|---|---|
| Wikipedia 文脈付き locked test | **主指標**。未見記事、reading 対照 pair、頻度 distractor | conditional Top-1、MRR/NDCG、pair 両方正解、層別値 |
| raw LM held-out documents | causal LM 学習の data scaling | tokenizer 非依存の bits-per-byte、各 tokenizer の NLL |
| AJIMEE 200 | 長文 end-to-end | CER / exact、fresh oracle recall。train overlap 除去必須 |
| Mozc 498 | 無文脈 regression | current 88/498 を悪化させない。left-only 条件を使用 |
| handmade 14 | `考証/交渉` 等の sentinel | Top-1 14/14、NDCG 0.992972、MRR 1.0 を維持 |
| Intel N100 benchmark | production feasibility | p50/p95、RSS、model bytes。quality 選定後の別 gate |

### 5.2 ranking と retrieval を分離する指標

全 decoder arm で次を同時に保存する。

1. `oracle recall@k`
2. `conditional Top-1@k`
3. recoverable-error recovery
4. baseline-correct → model-wrong regression rate
5. MRR@k / NDCG@k
6. end-to-end CER / exact
7. context contrast pair の両方正解率
8. frequency bucket、ambiguity 数、span 長、文脈長、固有名詞別の値
9. candidate ごとの decoder sum logprob、normalized score、Viterbi cost、gold rank
10. peak allocated / reserved VRAM、tokens/s、step/s、wall time
11. N100 の cold / warm、p50 / p95、RSS、model bytes

候補外 miss は conditional ranking の分母から外すが、end-to-end CER からは外さない。

### 5.3 統計と再現性

- corpus revision、split manifest、dictionary SHA、candidate artifact SHA、model/config SHA、dependency lock、seed を保存する。
- pilot は1 seed、採用判断は3 seed。
- model 間は同じ test item の paired bootstrap 95% CI。seed ごとの値と min/median/max も隠さない。
- hyperparameter、length normalization、Viterbi blend は dev のみで選ぶ。
- test set は D1 の実行前に凍結する。
- S200 と Qwen の tokenizer が異なるため perplexity の大小を直接比較しない。bits-per-byte と downstream 精度を使う。

---

## 6. RTX 3060 Ti 実行環境と VRAM 設計

### 6.1 machine topology

- Hermes（Intel N100 miniPC）と RTX 3060 Ti machine は別 machine。
- training / checkpoint 作成は RTX 3060 Ti machine で行う。
- Hermes は data artifact の準備・保管、最終 N100 inference benchmark に使えるが、training process を Hermes 上で走らせない。
- RTX machine へ Tailscale 経由 SSH で入る。Tailscale、Windows OpenSSH、WSL sshd の設定はユーザー側の前提作業であり、本計画では変更しない。
- SSH 切断で run を失わないよう `tmux` 等の persistent session を使い、checkpoint を定期保存する。artifact は SHA-256 付きで Hermes 側へ戻す。

### 6.2 OS の選択

**推奨: Windows host + WSL2 Ubuntu + CUDA-enabled PyTorch。**

理由:

- Linux 用 ML package / script と Colab の実行環境を揃えやすい。
- NVIDIA が WSL2 CUDA workflow を公式に提供している。
- bitsandbytes は Linux / Windows の両方を支援するが、Colab へ移る時の差分は Linux の方が小さい。

手順上の注意:

1. Windows 側に対応 NVIDIA driver を入れ、WSL2 と Ubuntu を有効化する。
2. WSL 内へ NVIDIA Linux display driver を入れない。CUDA toolkit が必要なら WSL-Ubuntu 用 package を使う。
3. repo、dataset cache、checkpoint は WSL の Linux filesystem に置く。Windows mount 越しの大量 small-file I/O は避ける。
4. Windows native は WSL2 で再現不能な blocker が出た場合の fallback とする。fallback 時は同じ dependency version、seed、artifact format を固定する。

SSH は次のどちらか一方に固定する。

- Windows OpenSSH へ入り、`wsl.exe -d <distro> -- bash -lc ...` で実行。
- WSL 内 sshd へ直接入り、Linux shell で実行。

経路を run ごとに変えず、hostname、Windows build、WSL kernel、driver、CUDA runtime を manifest に残す。

### 6.3 framework

- PyTorch
- Hugging Face Transformers / Datasets / Tokenizers
- Accelerate
- PEFT
- bitsandbytes（D1/D2 の QLoRA、必要時の 8-bit optimizer）
- PyTorch SDPA を既定 attention backend とし、FlashAttention は install 成功だけを前提にしない。

実行開始時に互換する version set を lock file と artifact manifest に固定する。`latest` のまま再実行しない。

### 6.4 最初に記録する probe

```text
nvidia-smi
python / torch / transformers / accelerate / peft / bitsandbytes versions
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)
torch.cuda.get_device_properties(0).total_memory
torch.cuda.is_bf16_supported()
1 forward/backward step for FP16 and BF16
```

Ampere 自体は BF16 Tensor Core を持つが、最初の recipe は **FP16 AMP**を既定にする。BF16 は probe が通り、100–1,000 step で NaN、速度、peak VRAM が FP16 以上なら採用する。`torch.cuda.is_bf16_supported()` の真偽だけで安定性を断定しない。

### 6.5 VRAM 算術と実測 gate

標準 mixed-precision AdamW を `[見積]` 16 bytes / trainable parameter（weight / master copy、gradient、FP32 first/second moments の代表的算術）と置く。PyTorch AdamW は first / second moment を持ち、`foreach` は追加 peak memory を使いうる。実装差があるため見積もりは予約値ではない。

| recipe | parameter 関連 `[見積]` | activation 等 | 8 GB 判断 |
|---|---:|---|---|
| 0.49B full training | 約7.84 GB（約7.30 GiB） | 必ず追加 | No-Go |
| 204M full training | 約3.26 GB（約3.04 GiB） | seq / batch / layers に依存 | checkpointing 込みで試せる |
| 0.49B LoRA FP16 base | frozen base 約0.98 GB + adapter states | activation が主 | 試せる |
| 0.49B QLoRA | raw 4-bit base 約0.25 GB + quant metadata / adapter states | activation / CUDA overhead が主 | 最初の recipe |

hard gate は算術ではなく `torch.cuda.max_memory_allocated/reserved` の実測とする。長時間 run は peak reserved **7.2 GiB 以下**を暫定 headroom とし、100–1,000 step、evaluation、checkpoint save を含めて OOM がないことを確認する。

### 6.6 batch / sequence 調整

初期値は最適値ではなく `[推測]`。

| run | sequence | micro-batch | accumulation | memory 手段 |
|---|---:|---:|---:|---|
| D1 task FT | 128 | 1 | effective batch を合わせるまで16–64 | NF4 QLoRA、gradient checkpointing、`use_cache=False` |
| D2 continued PT | 256 | 1 | token batch を固定 | NF4 QLoRA、packing、checkpointing |
| S200 scratch smoke | 256 | 1 | token batch を固定 | FP16/BF16、checkpointing、必要なら AdamW `foreach=False` |

調整順:

1. micro-batch 1、gradient accumulation で target token batch を作る。
2. activation checkpointing と SDPA を有効化する。
3. OOM なら sequence 256→192→128。task prompt が切れる率を同時に報告する。
4. それでも OOM なら S200 の depth / hidden を縮め、150M 未満に落ちる場合は比較 arm を変更したと明記する。
5. headroom があれば sequence を先に戻し、その後 micro-batch を増やす。
6. gradient accumulation は peak VRAM を micro-batch 1 より下げない。速度対策と memory 対策を混同しない。

各 run の最初の1,000 step から tokens/s と peak VRAM を取り、

$$
\text{projected hours}=\frac{\text{planned training tokens}}
{\text{measured tokens/s}\times3600}
$$

で完走時間を更新する。GPU 名から所要時間を推測しない。

---

## 7. 段階的実験と gate

### M0: data / evaluation gate（GPU 長時間利用なし）

- #6 後 baseline、candidate sets、oracle recall を machine-readable artifact として固定。
- Wikipedia を article ID 付きで再生成し、評価 set 重複を除去。
- 10k / 1k / 2k task group を作成し、200 group 目視 audit。
- raw corpus を Qwen tokenizer と S200 tokenizer で数え直す。
- D0 と Viterbi の locked baseline を保存。

**No-Go:** label 正解率98%未満、gold-in-candidate の有効 train group が10k未満、split / near-duplicate leakage が残る。ここで Colab を買わない。

### M1: 3060 Ti 最初の 1 run

- arm: **D1 Qwen2.5-0.5B base QLoRA task FT**
- data: train 10k / dev 1k / test 2k、article disjoint
- objective: answer token + EOS の causal LM loss
- initial config: `[推測]` NF4、LoRA rank 16、seq 128、micro-batch 1、1 epoch、1 seed
- compare: Viterbi、D0、D1。sum / normalized candidate score は dev で一つに固定
- record: first 1,000 step の VRAM、tokens/s、NaN、checkpoint / resume、全評価指標

この run の問いは一つだけ:

> 8 GB の 0.5B decoder adapter が、同音候補で raw frequency より context を選ぶ signal を示すか。

**M1 Go:** 

- warm 1,000 step と evaluation を peak reserved 7.2 GiB 以下、OOM / NaN なしで通す。
- Wikipedia dev conditional Top-1 が D0 と Viterbi の双方より +3 points 以上。
- context pair 両方正解率が baseline より改善。
- baseline-correct regression 5%以下、high-frequency distractor subset が全体より5 points超悪化しない。

1 seed のため M1 Go は本採用ではない。

**M1 failure の分岐:** 

- OOM: checkpointing、seq、backend を調整して同一 run を一度再測定。data / objective が正常なら hardware-bound と分類。
- loss は下がるが task 精度が伸びない: objective / candidate score / label を診断。**Colab へ進まない。**
- train だけ伸び dev が悪化: data-limited / overfit。data を増やし、model を大きくしない。
- signal あり: M2 へ進む。

### M2: 3060 Ti scratch feasibility と data scaling

1. #4 100 MiB で S200 の tokenizer、packing、checkpoint、resume を smoke。
2. 100–1,000 step で peak VRAM / tokens/s を測る。
3. pinned Wikipedia を拡張し、少なくとも3つの累積 token horizon で held-out bits-per-byte と IME dev accuracy を記録する。
4. 100 MiB run は quality decision に使わない。
5. data を増やしても held-out bpb / task accuracy が伸びない場合、architecture 拡大や Colab の前に data quality / objective を見直す。

### M3: local scratch 対 FT 比較

- D1: 0.5B QLoRA SFT
- D2: 同一 raw docs で 0.5B QLoRA continued pretraining → 同一 SFT
- S200: 同一 raw docs で scratch pretraining → 同一 SFT
- exploration は1 seed。明確に劣る recipe を落とし、best scratch と best FT を3 seed で再実行。
- raw exposure は同じ document set とし、native token 数、FLOPs、wall time を併記。
- 3.4 の採用規則で精度の勝者を決める。

### M4: 有料 Colab / S500 gate

次を**全て**満たす場合だけ、有料 Colab で 0.5B scratch を実施する。

1. M1 で decoder task objective に signal がある。
2. M2 の複数 data horizon で S200 の held-out bpb と IME dev accuracy が data 増加に対して改善し、scratch の失敗が単なる starvation と判断できる。
3. S200 が best FT に接近、勝利、または parameter/data scaling で逆転可能とする実測根拠がある。`[推測]` だけで進めない。
4. provenance を確定した十分な corpus がある。暫定 floor 1B unique native token 未満なら、その量で進む根拠を M2 curve から示す。10B reference との差を明記する。
5. 3060 Ti の1,000-step tokens/s と予定 token 数から S500 compute を算出し、Colab の割当 GPU で100–1,000 step benchmark を取り直す。
6. 割当 GPU の実測 peak reserved が VRAM の80%以下。標準 full-parameter AdamW が入らなければ、recipe 変更（8-bit optimizer、seq 短縮）を記録し、D2 との optimizer 差を明記する。
7. 1 checkpoint interval が Colab termination より十分短く、6時間以下を初期目標にする。resume test を完了し、VM 消失で失うのを1 interval 分に限定する。
8. account に表示された compute unit / 料金と projected runtime をユーザーが受容できる。

**Colab 実行時:** paid plan 名から GPU を仮定しない。接続ごとに GPU 型、VRAM、CUDA、tokens/s を記録する。通常12時間、Pro+ 最大24時間という上限を前提に、単一の連続 run へ依存しない。checkpoint、optimizer、scheduler、RNG、data cursor を保存して再開する。

**No-Go:** data / objective failure、1B未満で scaling 根拠なし、割当 GPU の memory / runtime 未達、checkpoint resume 不成立。これらは課金で解決しない。

### M5: 副次 encoder CE

M3 後に必要な場合だけ旧版の30M listwise FTを実施する。主に次を答える。

- decoder と同じ task data で encoder の conditional Top-1 はどこまで出るか。
- N100 で decoder より実用的か。
- decoder の gain は言語モデル事前学習によるものか、task format 自体によるものか。

### M6: production gate

quality winner を選んだ後だけ量子化 / runtime を比較する。

- candidate k=5 whole-batch、tokenize + inference + score blend を含む。
- Intel N100 warm p95 50 ms / 変換全体以下を hard gate、20 ms 以下を stretch とする。
- AJIMEE / Mozc / handmade / Wikipedia locked test を再実行。
- decoder が quality winner でも latency 未達なら、非同期化をこの issue に混ぜず、encoder CE / cascade または別 issue とする。

---

## 8. Go / No-Go 基準まとめ

### data Go

- 200 group audit で label 正解率98%以上。
- article-disjoint、evaluation overlap 除去済み。
- gold-in-candidate の有効 train group 10k以上。
- source revision / license / fingerprint manifest 完備。

### decoder signal Go

- M1 dev conditional Top-1 が D0 / Viterbi より +3 points 以上。
- context contrast 改善、regression 5%以下、frequency bucket 崩壊なし。
- 8 GB で安定し、resume できる。

### scratch / FT 採用 Go

- locked Wikipedia test の paired 95% CI が勝者側で 0 を跨がない。
- 3 seed 全て同方向。
- handmade 14/14、Mozc 88/498 以上。
- AJIMEE CER が fresh baseline を悪化させない。改善 / 悪化は paired CI と具体例を報告。
- retrieval miss を除いた指標と end-to-end 指標の両方を提示。

### Colab Go

- decoder signal、S200 data scaling、corpus、実測 runtime、実測 VRAM、resume、予算の8条件を全て満たす。
- 「3060 Ti でいまいち」の意味が **capacity / compute bound** と実測で分類されている。data / objective failure は Colab No-Go。

### production Go

- quality winner が N100 warm p95 50 ms / 変換全体以下。
- model / data license、attribution、再生成 manifest 完備。

---

## 9. リスク

| リスク | 検出 | 対策 |
|---|---|---|
| #4 だけで scratch を評価し誤って棄却 | token/parameter、data curve | 100 MiB は smoke 限定。複数 horizon で判断 |
| scratch が corpus を暗記 | train/held-out bpb gap、near-duplicate | unique docs 拡張、dedup、article split |
| pretrained FT が prior 頻度へ戻る | `交渉/考証`、frequency bucket、context pair | reading-balanced SFT、hard negatives、D1/D2 分離 |
| tokenizer 差で perplexity を誤比較 | native token 数の差 | bits-per-byte と task accuracy |
| candidate length bias | sum/mean score 差 | EOS を含め、dev で normalization 固定 |
| gold が候補にない | oracle recall@k | ranking / retrieval 分離。model を大きくしない |
| Wikipedia / AJIMEE leakage | article ID / fingerprint | evaluation 文書除外、locked manifest |
| 自動かな化の誤 label | 200 group audit | 98% gate、多音語 / 固有名詞 bucket |
| 200M full training OOM | 1,000-step peak VRAM | checkpointing、micro-batch 1、seq 調整 |
| 0.5B full scratch OOM | parameter-state 算術と実測 | 8 GB では実施しない。Colab gate |
| QLoRA と scratch の optimizer 差 | recipe manifest | 実用比較と明記。S500/D2 で初期化比較 |
| SSH / Colab 中断 | resume drill | checkpoint、RNG / cursor 保存、短い interval |
| Colab GPU が期待と違う | 接続後 probe | paid plan 名を信用せず実測 gate |
| test 過適合 | locked test、3 seed、CI | dev 以外で tuning しない |
| N100 latency 超過 | whole-conversion p95 | quality 後に量子化、encoder CE を副次比較 |
| corpus license 不明 | manifest / dataset card | CC-100 を初期対象外 |

### 再試行しない方向

旧版の repo 実測を引き継ぎ、次は新しい仮説なしに繰り返さない。

1. Qwen2.5-0.5B の平均 token 尤度をそのまま自然さとする。
2. Wikipedia raw 頻度だけを強める。
3. candidate recall を測らず model size だけ上げる。
4. #4 100 MiB を本格 scratch corpus と呼ぶ。
5. 8 GB で 0.5B standard AdamW full scratch を開始する。
6. data / objective failure を Colab 課金で解決しようとする。
7. paid Colab なら特定 GPU / 連続時間が保証されると仮定する。
8. AJIMEE 200件を見ながら hyperparameter を調整する。
9. CC-100 を「公開 data」だけを理由に投入する。

---

## 10. 実行チェックリスト

### 3060 Ti 前

- [ ] M0 の baseline / candidates / split を凍結
- [ ] 200 group audit と leakage 除去
- [ ] WSL2 CUDA と SSH 経路を実機 probe
- [ ] dependency version を固定
- [ ] FP16 / BF16 1-step probe
- [ ] checkpoint / resume dry run

### 最初の 1 run

- [ ] D1 Qwen2.5-0.5B QLoRA、10k group、1 seed
- [ ] 1,000-step VRAM / tokens/s / NaN 記録
- [ ] Viterbi / D0 / D1 を同じ candidates で比較
- [ ] context pair / frequency bucket / regression を出力
- [ ] M1 Go / failure class を記録

### local 比較

- [ ] #4 を smoke 限定で S200 に流す
- [ ] raw corpus の複数 token horizon を作る
- [ ] D1 / D2 / S200 を同一 protocol で1 seed比較
- [ ] best scratch / best FT を3 seed比較
- [ ] paired CI と全評価 suite を出す

### Colab 判断

- [ ] M4 の8 gate を全て確認
- [ ] account 表示価格 / compute unit を確認
- [ ] 割当 GPU で100–1,000 step 再 benchmark
- [ ] 6時間以下の checkpoint interval と resume を確認
- [ ] S500/D2 の差分を initialization 以外も含めて記録

### production 判断

- [ ] quality winner を確定、または精度同等と明記
- [ ] N100 whole-conversion p50/p95 / RSS / bytes
- [ ] AJIMEE / Mozc / handmade / locked Wikipedia 再評価
- [ ] data / model provenance と attribution 方針を確認

---

## 11. 一次情報

### repo / issue

- [Issue #5](https://github.com/sarashin65/beme/issues/5)
- [Issue #4](https://github.com/sarashin65/beme/issues/4)
- [PR #6](https://github.com/sarashin65/beme/pull/6)
- [BeME spec](../../spec.md)
- [deepdive](../../deepdive-2026-08-21.md)
- [P1 offline experiment](../../p1-offline-experiment-2026-08-21.md)
- [P2 k expansion report](../../p2-kexp-report.md)
- [#4 Phase 1 report](../../issue4-phase1-report.md)
- [P0 後 oracle recall](../../oracle-recall-after-p0-2026-08-21.md)
- [旧 PLAN](./PLAN.md)

### model / training / hardware

- [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B) — base causal LM、0.49B、architecture、license、Japanese support
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — parameter / training token scaling、Table 3
- [QLoRA](https://arxiv.org/abs/2305.14314) — frozen 4-bit base + LoRA、NF4、double quantization、paged optimizer
- [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/main/en/index) / [installation](https://huggingface.co/docs/bitsandbytes/main/en/installation) — QLoRA、8-bit optimizer、OS / CUDA / compute capability support
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — first / second moment、`foreach` peak memory 注意
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html) — compute / memory trade-off
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) — FP16 / BF16 Tensor Core
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) — WSL2 CUDA setup、Windows driver、Linux driver 禁止
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) — dynamic GPU / limits、12h / Pro+ 24h runtime

### data / license

- [Wikimedia Wikipedia dataset card](https://huggingface.co/datasets/wikimedia/wikipedia)
- [Wikimedia Terms of Use §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content)
- [CC-100 dataset card](https://huggingface.co/datasets/statmt/cc100)
