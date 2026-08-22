# 分割誤りの内訳分析 — AJIMEE 200 件

**測定日**: 2026-08-22  
**対象ブランチ**: `main`  
**対象コミット**: `06c612126fbef5f4b35a816710c74b455c09ebb4`  
**評価辞書**: `data/dictionary_learned.tsv` 349,043 エントリ、SHA-256 `541064d361a66e31290eeea1008b852f3453410ab48077d9ab860f03586afd35`  
**評価集合**: `data/ajimee_bench_jwtd_v2_v1.json` 全 200 件、SHA-256 `e9eb668fd6aa14b1e26436f429b5550108af0a1dfd443b8cea0bcb3ab3028fca`  
**共通条件**: `beam_width=64`、現行 whole-sentence lattice、`MockContextReranker` の keyword を空にして Phase 1 の候補順を測定  
**分析スクリプト**: `~/.cache/beme-diag/split_error_analysis.py`  
**生データ**: `~/.cache/beme-diag/split-error-analysis.json`  

## 1. 結論

分割誤り 100 件の主像は、単純な「機能語を長い語へ呑み込む過連結」ではなかった。
純粋な過分割が 37 件、純粋な過連結が 9 件、余分な境界と欠けた境界を同時に持つ境界ずれ・混合が 54 件だった。
100 件中 91 件に余分な境界があり、余分な境界 153 箇所のうち 109 箇所は正解側 noun token の内部、30 箇所は正解側 verb token の内部だった。
case 単位では noun または verb 内の過分割が 88/100 件に現れた。

この過分割を駆動している第一の信号は lexical cost の不均衡である。
余分な境界 153 箇所のうち 114 箇所、77 case では境界の左右どちらかが `dictionary_manual.tsv` の overlay entry だった。
129 箇所、81 case では少なくとも片側 token の cost が IPADIC floor 1.75 未満だった。
典型は正解 `外部` を `が(0.1) + 慰ぶ`、`保健科` の一部を `か(0.1)`、`白` を `し(0.1) + 炉` のように切る経路である。
接続コスト欠落だけでなく、欠落を補償するために安くした一文字 overlay が内容語内部でも強いアトラクタになる補償関係が、分割誤りの大半へ波及している。

一方、過連結側にも `matrix.def` の信号は残る。
正解側にあるのに 1-best から消えた境界は 99 箇所で、56 箇所の IPADIC 接続コストが負だった。
接続コストの平均は raw −1,622、中央値 −557、BeME 換算では平均 −0.811、中央値 −0.279 だが、現行 coarse POS transition は同じ 99 箇所へ平均 +0.085 の罰を置く。
特に接尾辞吸収 9 箇所は全件が負で、平均 raw −4,463、助動詞吸収 9 箇所は 7 箇所が負だった。
ただし機能語吸収は particle・auxiliary・suffix の和で 30 event、重複を除くと 26/100 case であり、100 件全体の単一原因ではない。

辞書欠落との厳密な複合はほぼ無い。
排他 DP 定義上、100 件の正解経路に patch edge は 0 件だった。
`ajimee-jwtd-v2-1429` の長音 `ー` だけは正解・仮説の双方で lattice unknown fallback を通るが、同じ文字を保持しており、観測された境界誤りの直接原因ではない。

文脈リランカの分割だけに対するオラクル上限は、現行 5-best で 20/100 件だった。
このうち現行 AJIMEE が供給する preceding `context_text` が非空なのは 12 件である。
`original_text` から following context を導出して両側文脈を利用可能とみなすと 20 件すべてに文脈がある。
候補幅を 64 へ広げると正解境界 skeleton は 41 件に存在し、前後いずれかの文脈もあるのは 40 件、preceding だけが実データにあるのは 25 件だった。
ただし正解全文そのものの候補包含は分割誤り集合で 5-best 3 件、64-best 11 件にすぎず、境界だけ直っても表記誤りは残る。

次の実験は、reading field を index 7 に直したコスト学習辞書の再生成と、current lexical cost を残す小重み `matrix.def` hybrid を優先する。
本分析の hybrid sweep では、現行 transition と `matrix.def/2000` の補間係数 α=0.25 が AJIMEE CER 0.277928 → 0.266171、差 −0.011757、exact 14 → 17 だった。
ただし同じ 200 件で α を比較した診断値であり、採用値ではない。
α=0.25 が正解境界へ直したのは 9/100 件だけなので、matrix 単独では overlay 起因の過分割を解消しない。

## 2. 定義と方法

### 2.1 分割誤り集合

`docs/p3-frequency-prior-experiment-2026-08-21.md` と同じ whole-sentence oracle DP を再実行した。
DP は `(reading position, reference position, previous POS)` を状態とし、patch 本数、次いで現行 lexical + transition cost を最小化する。
現行 1-best と正解文字列を生成する最小 cost path の reading 境界集合を比較した。

再現した排他分解は次のとおりである。

| 排他分類 | 件数 | 全 200 件比 |
|---|---:|---:|
| モデル誤り・分割 | **100** | **50.0%** |
| モデル誤り・同音／表記選択 | 58 | 29.0% |
| 辞書欠落 | 25 | 12.5% |
| 正解 | 14 | 7.0% |
| 探索／完全同点 | 3 | 1.5% |

100 件の mean CER は 0.340175 だった。
全 AJIMEE CER error mass に対する寄与は既報どおり 61.2% である。

ここでいう「正解境界」は、人手で付与した文節・形態素境界ではない。
正解文字列を生成できる辞書 path のうち現行 cost が最小の path の境界である。
したがって、本レポートの「過分割」「過連結」は辞書 edge 境界の比較であり、言語学的な文節評価ではない。

### 2.2 下位分類

1-best の内部境界集合を $H$、oracle path の内部境界集合を $O$ とした。

- **過分割**: $H \setminus O \ne \varnothing$ かつ $O \setminus H = \varnothing$。
- **過連結**: $O \setminus H \ne \varnothing$ かつ $H \setminus O = \varnothing$。
- **境界ずれ・混合**: 両方が非空。

文末境界は両 path に必ずあるため比較から除外した。
句読点などの literal edge は whole-sentence lattice の token として両 path に保持した。

## 3. 分割誤り 100 件の内訳

### 3.1 case 単位の排他分類

| 下位分類 | 件数 | 100 件比 | 読み方 |
|---|---:|---:|---|
| **境界ずれ・混合** | **54** | **54%** | 余分な境界と欠けた境界を同時に持つ |
| **過分割** | **37** | **37%** | 正解側 1 token を仮説が 2 token 以上へ割る |
| **過連結** | **9** | **9%** | 正解側 2 token 以上を仮説が 1 token へ繋ぐ |
| 合計 | 100 | 100% | 排他的 |

欠けた境界を持つ case は 63 件、余分な境界を持つ case は 91 件だった。
欠けた境界は合計 99 箇所、余分な境界は合計 153 箇所だった。
欠けた境界を複数持つ case は 24 件、余分な境界を複数持つ case は 41 件だった。

この結果は「分割誤り」を token 数だけで説明できないことを示す。
54 件は一方を割りすぎながら別の場所を繋ぎすぎており、局所 lexical attraction が境界を移動させている。

### 3.2 余分な境界 — 過分割側の正体

| 正解側 token の内部 | event 数 | case 数 |
|---|---:|---:|
| noun 内 | **109** | **79** |
| verb 内 | 30 | 22 |
| adjective 内 | 5 | 3 |
| その他 | 8 | 8 |
| noun 列への過分割 | 1 | 1 |
| 合計 | 153 | 重複あり |

noun または verb 内の過分割は重複を除いて 88/100 case に現れた。
余分な仮説境界の coarse POS pair 上位は次のとおりだった。

| 仮説側 POS pair | event 数 | case 数 |
|---|---:|---:|
| **particle → noun** | **45** | **41** |
| noun → noun | 23 | 21 |
| noun → particle | 10 | 8 |
| particle → particle | 8 | 7 |
| noun → verb | 5 | 5 |
| particle → number | 5 | 5 |
| particle → adjective | 4 | 4 |
| verb → noun | 4 | 4 |
| adjective → noun | 4 | 4 |
| interjection → particle | 4 | 4 |

particle → noun が 45 箇所で最大なのは、正解内容語の先頭や中間を `か`、`し`、`と`、`が` などの安い一文字 particle として切るためである。
余分な境界の左右に出た cost 1.75 未満 token の上位は `か/か/particle=0.1` 19 回、`と/と/particle=0.1` 14 回、`し/し/particle=0.1` 13 回、`き/気/noun=1.4` 12 回、`も/も/particle=0.1` 9 回だった。

| overlay / 低 cost との隣接 | event 数 | case 数 |
|---|---:|---:|
| `dictionary_manual.tsv` entry が境界左右のどちらかにある | **114/153** | **77/100** |
| cost が IPADIC floor 1.75 未満の token が境界左右のどちらかにある | **129/153** | **81/100** |

隣接は因果そのものの証明ではないが、現行 cost が選ぶ余分な境界の大半が overlay の安価な token に接しているという直接観測である。
`docs/deepdive-2026-08-21.md` が示した「overlay は機能語接続報酬を lexical cost で肩代わりする」という補償が、内容語内部では逆向きの副作用を持つ。

### 3.3 欠けた境界 — 過連結側の正体

欠けた 99 境界の coarse POS pair 上位は次のとおりだった。

| 正解側 POS pair | event 数 | case 数 |
|---|---:|---:|
| **noun → noun** | **21** | **18** |
| particle → noun | 10 | 9 |
| noun → particle | 7 | 7 |
| noun → suffix | 6 | 6 |
| verb → auxiliary | 6 | 6 |
| number → number | 5 | 4 |
| verb → noun | 4 | 4 |
| number → noun | 4 | 4 |
| verb → verb | 4 | 4 |
| adjective → noun | 3 | 3 |
| verb → particle | 3 | 3 |
| prefix → noun | 3 | 3 |

IPADIC 細分類の上位は `名詞/一般 → 名詞/一般` 12 event / 10 case、`名詞/一般 → 名詞/接尾/一般` 5/5、`名詞/数 → 名詞/数` 5/4、`名詞/サ変接続 → 名詞/一般` 5/5、`助詞/接続助詞 → 名詞/副詞可能` 5/5 だった。
coarse `noun → noun` だけでは「普通名詞の複合」「数詞列」「サ変名詞」「接尾」を区別できず、deepdive の coarse POS R²=0.21 と整合する。

機構別に見ると次のとおりだった。

| 欠けた境界の機構 | event 数 | case 数 |
|---|---:|---:|
| 名詞列の過連結 | 24 | 20 |
| 助詞の吸収 | 12 | 11 |
| 動詞境界の過連結 | 10 | 9 |
| 接尾辞の吸収 | 9 | 9 |
| 助動詞の吸収 | 9 | 9 |
| 接頭辞の連結 | 3 | 3 |
| その他 | 32 | 24 |

particle・auxiliary・suffix の吸収を case union で数えると 26/100 件だった。
右側 token の頻出 reading/surface は `な/な/particle` 6 event / 5 case、`おり/おり/noun` 5/5、`れ/れ/auxiliary` 3/3、`て/て/particle` 3/3、`だ/だ/auxiliary` 3/3 だった。
局所表層パターンでは `て + おり → 手織り` が 5 case、`書く + こと → 枸杞` が 2 case、`的 + な → 規那` が 2 case、`よう + な → ウナ` が 2 case だった。

### 3.4 接続コストの符号

| 指標 | `matrix.def` raw | BeME 換算 `/2000` | 現行 coarse transition |
|---|---:|---:|---:|
| 欠けた境界 99 箇所の平均 | −1,622 | −0.811 | +0.085 |
| 中央値 | −557 | −0.279 | — |
| 最小 | −10,196 | −5.098 | — |
| 負の event | 56/99 | 56/99 | 0/99 |

正解境界の過半数では `matrix.def` が接続報酬を持つのに、現行表は正の罰だけを足している。
ただし 43/99 境界は `matrix.def` でも 0 以上であり、すべての過連結を matrix 欠落へ帰属させるのは誤りである。

機能形態素別では接尾辞吸収 9/9 が負、平均 raw −4,463、助動詞吸収 7/9 が負、平均 raw −1,667 だった。
助詞吸収は負が 5/12 にとどまり、平均 raw −1,663 だが中央値は +234 だった。
「て」系を含め、文脈または lexical frequency が必要な case が混ざる。

### 3.5 辞書欠落との複合

| 指標 | 件数 |
|---|---:|
| 正解 oracle path に patch edge が必要 | **0/100** |
| 仮説 path に lattice `unknown` がある | 1/100 |
| 正解 path に lattice `unknown` がある | 1/100 |

該当する 1 件は `ajimee-jwtd-v2-1429` で、`ドリブラー` の長音 `ー` を一文字 unknown fallback がそのまま出力する。
正解・仮説の双方が同じ fallback を通るため、境界差は `ド + リ` 対 `どり` や `体` 対 `から + だ` にあり、`ー` の辞書欠落は非因果的な複合である。

したがって、分割誤り集合に対する一般的な未知語追加は優先レバーではない。
これは排他 DP の定義に依存する結論であり、正解文字列を複数の既知 edge で綴れるが望ましい複合語 entry が無いケースまでは「辞書欠落」と数えない。

## 4. 代表例

### 4.1 純粋な過分割 — `ajimee-jwtd-v2-965`

```text
入力  がいぶのかしゅのがっきょくせいさくでは
正解  外部の歌手の楽曲制作では
出力  が慰ぶの歌手の楽曲政策では
境界  正解: 外部
      出力: が(particle, 0.1) + 慰ぶ(noun)
```

正解 `外部` 内部へ余分な `particle → noun` 境界を置く。
cheap particle が内容語先頭を奪う、今回最大の過分割パターンである。

### 4.2 純粋な過連結 — `ajimee-jwtd-v2-1893`

```text
入力  ぞうていしたふなもりの
正解  贈呈した船盛りの
出力  贈呈した舟守の
境界  正解: 船(noun) + 盛り(noun)
      出力: 舟守(noun)
```

正解 `船 + 盛り` の境界が消え、同じ reading を持つ長い名詞 `舟守` が勝つ。
この境界の `matrix.def` cost は raw +62 であり、matrix の負報酬だけでは直らない lexical・表記選択寄りの例である。

### 4.3 境界ずれ — `ajimee-jwtd-v2-1800`

```text
入力  にんむではかれからむせんでしじをうけて
正解  任務では彼から無線で指示を受けて
出力  任務ではかレから無線で指示を受け手
余分  彼 → か(particle) + レ(noun)
欠落  受け(verb) + て(particle) → 受け手(noun)
```

同じ文で過分割と過連結が同時に起きる。
`受け → て` の `matrix.def` cost は raw −308 だが、deepdive の検算どおり `受け手` は IPADIC 目的関数でも強く、matrix だけでは十分でない。

### 4.4 複数の cheap fragment — `ajimee-jwtd-v2-7`

```text
正解  青年誌で書くことの理由として、男性青年誌における男性優位の性意識（
出力  青年しでか枸杞との理由として、男性青年しにおける男性優位の誠意し気（
欠落  書く + こと → 枸杞
余分  書く → か + 枸杞、こと → 枸杞 + と
欠落  性 + 意識 → 誠意
余分  意識 → 誠意 + し + 気
```

一つの長語選択だけでなく、安い一文字 edge が前後へ侵入して複数境界を置換する。
この種が 54 件の境界ずれ・混合を作る。

## 5. 文脈で解ける件数のオラクル見積もり

### 5.1 定義

「文脈で解ける」のオラクル条件を、候補集合に oracle path と完全一致する reading 境界 skeleton が一つ以上あることとした。
perfect reranker がその候補を識別できるという上限であり、実リランカの達成値ではない。
surface は一致条件に入れていないため、分割だけの上限である。

following context は評価入力として直接保存されていない。
本分析では `original_text` 内に `expected_output` を定位し、その直後を following として導出した。
これは可用性の見積もりにだけ使い、candidate score には使っていない。
100 件中 preceding `context_text` が非空なのは 46 件、following を導出できたのは 98 件、前後いずれかがあるのは 99 件だった。

### 5.2 候補幅別上限

| 候補幅 | 正解境界 skeleton | preceding あり | following あり | 前後どちらかあり | 正解全文候補 |
|---:|---:|---:|---:|---:|---:|
| 5 | **20** | **12** | 20 | 20 | **3** |
| 10 | 25 | 17 | 24 | 25 | 3 |
| 20 | 32 | 21 | 31 | 32 | 8 |
| 64 | **41** | **25** | 39 | **40** | **11** |

現行固定 5-best と現行 AJIMEE preceding context だけを厳格に使う場合、分割を選び直せる上限は **12/100 件**である。
API が実装済みの following context も呼び出し側から供給する場合、同じ 5-best の上限は **20/100 件**である。
候補幅を 64 へ広げ、前後どちらかを供給する実験の上限は **40/100 件**である。

ただし正解全文候補は 5-best 3 件、64-best 11 件だけである。
したがって「文脈で分割を選べる 20/40 件」を、そのまま exact match の改善件数として扱ってはいけない。
表記選択を同時に直す Cross-Encoder の実回収率は別途測定が必要である。

### 5.3 ②の文脈リランカ実測への gate

次の実測は `k=5/20/64` を分け、preceding-only と preceding+following を分けるべきである。
主要観測値は exact match だけでなく、oracle boundary skeleton の回収数、CER、候補当たりレイテンシとする。
固定 5-best・preceding-only で 12 件の上限を追うより、まず 64-best・両側文脈で 40 件のうち何件を実回収できるかを見るほうが Phase 2 の判断材料になる。
一方、正解全文 oracle が 11 件しかないため、候補生成を直さず Cross-Encoder のみを production 最優先にする根拠はない。

## 6. `matrix.def` hybrid の追加診断

### 6.1 目的関数

既報 P1 の「IPADIC word cost + matrix へ全面切替」と異なり、今回は現行 learned lexical cost を保持した。
境界 cost だけを次式で補間した。

$$
C_{\mathrm{transition}}(\alpha)
= (1-\alpha) C_{\mathrm{current\ POS}}
+ \alpha \frac{C_{\mathrm{matrix}}}{2000}.
$$

IPADIC 由来 349,017 entry の left/right ID を原本 CSV から復元した。
overlay 21 entry は既報 P1 と同じ IPADIC macro の外側 ID を使い、unsupported overlay 5 entry は HIRAGANA unknown ID を使った。
literal の前後では IPADIC BOS/EOS を入れた。
α=0 の 200/200 output が公式 baseline と一致することを assertion で確認した。

### 6.2 sweep

| α | mean CER | baseline 差 | exact | 元の分割誤りで正解境界へ変化 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.277928 | ±0.000000 | 14/200 | 0/100 |
| 0.10 | 0.271820 | −0.006109 | 16/200 | 4/100 |
| **0.25** | **0.266171** | **−0.011757** | **17/200** | 9/100 |
| 0.50 | 0.269596 | −0.008332 | 16/200 | 15/100 |
| 0.75 | 0.288041 | +0.010113 | 19/200 | 16/100 |
| 1.00 | 0.327749 | +0.049821 | 17/200 | 16/100 |

α=0.25 は同一集合上で CER 最良だったが、正解境界への変化は 9 件だけだった。
この 9 件のうち現行 5-best にも正解境界があるのは 6 件、64-best にあるのは 8 件で、matrix と候補幅拡大の回収集合は大きく重なる。
α=0.50 は 15 件の境界を直すが aggregate CER は α=0.25 より悪く、境界数最大化と文字品質が一致しない。
α=1.00 は prior P1 の全面 matrix と同じ方向へ大きく悪化し、matrix を強く掛ければよいという仮説を再び否定した。

この sweep は同じ AJIMEE 200 件で比較した in-sample 診断である。
α=0.25 を production 値として採用してはならず、held-out gate が必要である。

## 7. 次に実装・実験すべきレバー

### 優先度 1 — reading field 修正後のコスト学習再生成と分割再測定

`scripts/apply_cost_learning.py:66` は IPADIC feature index 6 の原形を reading として使っており、正しい reading は index 7 である。
現在の `dictionary_learned.tsv` は漢字語の頻度を大半取り込めていない。
一方、今回の最大機構は 77〜81 case に現れる cheap overlay / low-cost fragment と、十分に下がっていない正解内容語 cost の競合である。

最初の gate は index 7 で frequency を再集計して学習辞書を repo 外に再生成し、AJIMEE の CER だけでなく本レポートの 37/9/54、extra 153、overlay 隣接 114 を再測定することである。
現行 `apply_costs` は頻出語の cost を下げるだけで誤 fragment を直接罰しないため、効果は未実測である。
再生成だけで不足する場合は、overlay を一括削除せず、機能語として必要な用法と内容語内部の fragment 用法を分ける lexical calibration を学習対象にする。
held-out 分割を使い、P0 の補償を壊さないことを gate にする。

### 優先度 2 — 小重み `matrix.def` hybrid の held-out 実験

現行 lexical cost を保持した α=0.10〜0.50 は全点で baseline CER を改善し、α=0.25 は −0.011757 だった。
これは既報 P1 の全面 IPADIC 目的関数が悪化した結果と両立し、matrix の局所信号だけを弱く回収する設計に実験価値がある。

実験では α、BOS/EOS、overlay macro の ID、P0 penalty を共同で変えず、一つずつ ablate する。
AJIMEE を tuning と評価に分離し、Mozc 498 件でも CER 相当・pass・レイテンシを測る。
α=1.0 と IPADIC word cost 全面移行は再試行しない。

### 優先度 3 — overlay / 一文字 fragment の discriminative calibration

余分な境界の 74.5% に当たる 114/153 event が manual overlay entry に隣接し、84.3% に当たる 129/153 event が IPADIC floor 未満 token に隣接した。
この信号は matrix より広い 77〜81 case を覆う。

ただし overlay 全削除は deepdive で CER +0.124 と実測済みなので禁止する。
候補は、正しい reading field の corpus count、token が長い dictionary edge 内部へ出る頻度、left/right context を使って一文字 overlay の cost を学ぶことである。
単純な token 数罰は 54 件の境界置換と機能語の必要境界を同時に壊すため避ける。

### 優先度 4 — 候補幅を広げた両側文脈リランカ PoC

固定 5-best の分割 skeleton 上限は両側文脈でも 20 件、正解全文は 3 件である。
64-best なら分割 skeleton 40 件、正解全文 11 件まで増えるため、文脈リランカを試すなら `k=20/64` と following context を同時に使う。

実装済み Cross-Encoder をそのまま使い、候補幅別の実回収率を測る。
40 件は perfect reranker の境界上限であり、期待改善値ではない。
lexical / hybrid gate より先に production 化しない。

### 優先度 5 — P3 tie prior は独立に継続

P3 frequency prior は分割 skeleton を固定するため、今回の 100 件を直接直さない。
一方、hybrid や文脈候補で境界が直った後に残る表記差を直す役割がある。
index 7 修正と held-out gate を条件に、既報どおり exact `(reading, POS, lexical cost)` tie に限定して進める。

### 優先しないもの

- `matrix.def` の full weight または IPADIC word cost への全面切替。
- 16 POS transition 表の手動調整。
- overlay の一括削除。
- ビーム幅だけの拡大。
- 分割誤り対策としての一般的な未知語辞書追加。
- 固定 5-best のままの Cross-Encoder production 投資。

## 8. 制約と解釈上の注意

- 分割境界は oracle DP が選ぶ辞書 path の境界であり、人手形態素境界ではない。
- case 数と event 数は別であり、pattern 表の case 数は相互に重複する。
- overlay 隣接率は強い相関だが、その token を除けば必ず正解になるという反実仮想実験ではない。
- `matrix.def` transition の fine POS は復元した代表 source record に基づく。
- 99 の欠落境界はすべて left/right ID へ対応できた。
- context oracle は perfect selection の上限であり、Cross-Encoder を実行した値ではない。
- following は `original_text` から導出したため、現在の AJIMEE loader が供給する情報ではない。
- hybrid α sweep は AJIMEE 200 件への in-sample 比較であり、汎化性能を示さない。
- production code と `src/` は変更していない。
- git commit / push は行っていない。

## 9. 再現手順と生成物

```sh
cd ~/work/beme
.venv/bin/python ~/.cache/beme-diag/split_error_analysis.py
# -> ~/.cache/beme-diag/split-error-analysis.json
```

リポジトリ内の生成物は `docs/split-error-analysis-2026-08-21.md` のみである。
分析スクリプトと全 100 件の token・boundary・context oracle・hybrid sweep 生データは `~/.cache/beme-diag/` に保存した。
