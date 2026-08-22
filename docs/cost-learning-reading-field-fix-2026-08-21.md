# コスト学習 reading field 修正・辞書再生成・分割再測定

**測定日**: 2026-08-22  
**対象ブランチ**: `main`  
**対象コミット**: `06c612126fbef5f4b35a816710c74b455c09ebb4`  
**修正対象**: `scripts/apply_cost_learning.py`  
**入力辞書**: `data/dictionary.tsv`、349,043 エントリ、SHA-256 `efcdf5a1baeaeb74f8ca6c85552d6fac335d40ad274f1e77181c21c8f37f617b`  
**旧学習辞書**: `data/dictionary_learned.tsv`、349,043 エントリ、SHA-256 `541064d361a66e31290eeea1008b852f3453410ab48077d9ab860f03586afd35`  
**新学習辞書**: `data/dictionary_learned_v2.tsv`、349,043 エントリ、SHA-256 `52a3b9affb68763b0dfa9f624d7d661ba81520699a59c1475a6ad0f5ace40990`  
**コーパス**: `data/corpus.txt`、104,857,600 bytes、SHA-256 `ffd11238de39225c52a5e81da85cf4d6722fd251ad5881d1c98c02af0d238b60`  
**評価集合**: `data/ajimee_bench_jwtd_v2_v1.json` 全 200 件、SHA-256 `e9eb668fd6aa14b1e26436f429b5550108af0a1dfd443b8cea0bcb3ab3028fca`  
**分割分析条件**: `beam_width=64`、`MockContextReranker(keywords={})`、whole-sentence lattice  
**分割分析生データ**: `~/.cache/beme-diag/split-error-analysis-v2.json`  

## 1. 結論

IPADIC feature index 6 の原形を reading としていたバグを、正しい index 7 の読みに修正した。

正しい field では集計量が 7,548,547 tokens / 21,829 unique から 15,349,607 tokens / 106,785 unique `(reading, surface, POS)` へ増え、ベース辞書のコスト変更行は 955 行から 4,995 行へ増えた。

新辞書は旧辞書に対して AJIMEE mean CER を 0.277928 から 0.274751 へ 0.003177 改善し、exact を 14/200 から 15/200、Mozc pass を 90/498 から 92/498 へ改善した。

Handmade は Top-1 と MRR@5 が不変で、NDCG@5 は 0.953679 から 0.943936 へ 0.009743 低下した。

分割誤りの総数は 100/200 のままで減らなかった。

一方、純粋な過分割は 37 件から 32 件、余分な境界は 153 箇所から 141 箇所、noun または verb 内の過分割 case は 88 件から 80 件へ減ったため、過分割側には限定的な改善がある。

その代わり純粋な過連結は 9 件から 16 件、欠けた境界は 99 箇所から 106 箇所へ増えており、分割問題を解消したのではなく過分割から過連結へ一部を移した結果である。

今後のコスト学習 baseline は、既知バグを含む `dictionary_learned.tsv` ではなく `dictionary_learned_v2.tsv` とする。

旧辞書は履歴比較専用として残し、分割誤り対策は v2 を起点に別途 lexical calibration または小重み transition hybrid を評価する。

## 2. reading field 修正

`fugashi.GenericTagger` と IPADIC の feature 配列を実測すると、`外部` は index 6 が原形 `外部`、index 7 が読み `ガイブ` だった。

修正前は index 6 をひらがな化していたため、漢字の原形が後段のひらがな検査で脱落していた。

修正後は `fields[7]` を使用し、index 7 へ安全にアクセスするため長さ条件も `len(fields) < 8` に直した。

修正後の単語単位 smoke test では `外部` が `('がいぶ', '外部', 'noun'): 1` として集計された。

## 3. 学習辞書の再生成

次のコマンドでベース辞書から新規生成し、旧学習辞書は上書きしていない。

```sh
.venv/bin/python scripts/apply_cost_learning.py \
  --input data/dictionary.tsv \
  --output data/dictionary_learned_v2.tsv \
  --json
```

生成結果は次のとおりだった。

```json
{"changed_entries":4995,"matched_entries":4995,"top_n":5000,"total_tokens":15349607,"unique_tokens":106785}
```

既存設定の `top_n=5000`、`alpha=0.1`、`min_cost=0.1` は変更していない。

バグ修正だけの効果を旧辞書と比較するため、同時に hyperparameter を変更すると原因を分離できないと判断した。

aggregate 評価は改善したものの分割誤り総数は減っておらず、同じ AJIMEE 200 件へ合わせた追加調整は in-sample tuning になるため、今回の再生成結果だけを根拠に設定を変更しない。

ベース辞書との差分は旧辞書 955 行、新辞書 4,995 行だった。

旧新の双方で変わった行は 945 行、旧だけで変わった行は 10 行、新だけで変わった行は 4,050 行で、旧新間の cost が異なる行は 4,994 行だった。

`.gitignore` は `data/dictionary_learned.tsv` だけを明示的に ignore しており、`data/dictionary_learned_v2.tsv` は ignore 対象ではないことを `git check-ignore -v` で確認した。

新辞書を本タスクの確認可能な成果物として残すため、`.gitignore` は変更していない。

## 4. 全評価スイート

両辞書へ同じ公式評価コマンドを実行した。

```sh
.venv/bin/python scripts/evaluate.py --suite all --dictionary data/dictionary_learned.tsv
.venv/bin/python scripts/evaluate.py --suite all --dictionary data/dictionary_learned_v2.tsv
```

| 指標 | 旧 `dictionary_learned.tsv` | 新 `dictionary_learned_v2.tsv` | 新 − 旧 |
|---|---:|---:|---:|
| Handmade NDCG@5 | 0.953679 | 0.943936 | −0.009743 |
| Handmade MRR@5 | 0.928571 | 0.928571 | ±0.000000 |
| Handmade Top-1 | 0.857143 | 0.857143 | ±0.000000 |
| AJIMEE mean CER | 0.277928 | **0.274751** | **−0.003177** |
| AJIMEE exact | 14/200 = 0.070000 | **15/200 = 0.075000** | **+1 件 / +0.5 points** |
| Mozc pass | 90/498 = 0.180723 | **92/498 = 0.184739** | **+2 件 / +0.4016 points** |
| Mozc skipped | 2 | 2 | 0 |

候補生成 latency の表示値は両方 0.002 ms だった。

## 5. 分割内訳の再測定

`~/.cache/beme-diag/split_error_analysis.py` の分類、oracle、境界集計ロジックを新辞書へ適用した。

元スクリプトには旧辞書の decomposition `100/58/25/14/3` を固定する assertion があり、新辞書の decomposition `100/57/25/15/3` を受け付けないため、再測定時だけ assertion を辞書非依存の exhaustiveness 検査へ置き換えた。

依頼対象外の `matrix.def` hybrid sweep は、新辞書で alpha=0 の独自 variant 復元が公式 baseline と 2 件一致しなかったため実行対象から外した。

本節の分類、oracle、extra boundary、overlay 隣接、低 cost 隣接の値は hybrid sweep より前に同じ公式 Converter 経路から集計され、最終 JSON へ保存された。

再測定後、cache 上の分析スクリプトは元の内容へ復元した。

### 5.1 排他分類と境界数

| 指標 | 旧辞書 | 新辞書 | 新 − 旧 |
|---|---:|---:|---:|
| 分割誤り | 100 | 100 | 0 |
| 純粋な過分割 | **37** | **32** | **−5** |
| 純粋な過連結 | **9** | **16** | **+7** |
| 境界ずれ・混合 | **54** | **52** | **−2** |
| 余分な境界 event | **153** | **141** | **−12** |
| 余分な境界を持つ case | 91 | 84 | −7 |
| 欠けた境界 event | 99 | 106 | +7 |
| 欠けた境界を持つ case | 63 | 68 | +5 |
| 余分な境界を複数持つ case | 41 | 35 | −6 |
| 欠けた境界を複数持つ case | 24 | 25 | +1 |
| 分割誤り case の mean CER | 0.340175 | 0.329891 | −0.010284 |

旧分割誤り集合から 7 件が外れ、別の 7 件が新たに分割誤りへ入り、総数は同じ 100 件になった。

したがって純粋な過分割 5 件減を「5 件の分割誤りを修正した」と解釈してはならない。

### 5.2 過分割パターン

| 指標 | 旧辞書 | 新辞書 | 新 − 旧 |
|---|---:|---:|---:|
| noun 内の余分な境界 event | 109 | 95 | −14 |
| verb 内の余分な境界 event | 30 | 29 | −1 |
| adjective 内の余分な境界 event | 5 | 3 | −2 |
| noun または verb 内の過分割 case union | 88 | 80 | −8 |
| manual overlay 隣接 event | **114** | **113** | **−1** |
| manual overlay 隣接 case | 77 | 70 | −7 |
| cost < 1.75 隣接 event | **129** | **121** | **−8** |
| cost < 1.75 隣接 case | 81 | 76 | −5 |

内容語 cost が正しい頻度で下がったことで noun 内の余分な境界は 14 箇所減り、主要な過分割パターンは改善した。

ただし余分な境界 141 箇所のうち manual overlay 隣接はなお 113 箇所、cost 1.75 未満隣接は 121 箇所であり、cheap overlay / low-cost fragment の attraction は支配的なままである。

分母も変わるため、overlay 隣接 event の比率は 114/153 = 74.5% から 113/141 = 80.1% へ上がった。

正しい reading field による頻出内容語の値下げだけでは cheap fragment を直接罰しないという事前分析の注意は、そのまま残る。

## 6. コーパスの出所とライセンス

`scripts/fetch_corpus.py` は Hugging Face の `wikimedia/wikipedia`、config `20231101.ja`、split `train` から先頭 100 MiB を取得する。

取得元の一次資料である [Wikimedia Wikipedia dataset card](https://huggingface.co/datasets/wikimedia/wikipedia) を 2026-08-22 に確認した。

同 card はデータを Wikipedia dumps 由来と説明し、license metadata を `cc-by-sa-3.0,gfdl` と明記している。

したがって本 `corpus.txt` の保守的な出所・ライセンス記録は、Japanese Wikipedia `20231101.ja` 由来、CC BY-SA 3.0 + GFDL である。

`corpus.txt` 自体は `.gitignore` 対象であり、本タスクでは再配布していない。

## 7. 判断と次の baseline

`dictionary_learned_v2.tsv` を今後の cost-learning baseline とする。

理由は、生成パイプラインの既知バグを除去し、AJIMEE CER、AJIMEE exact、Mozc pass を同時に改善し、Handmade Top-1 を維持したためである。

ただし Handmade NDCG@5 の小幅低下と、分割誤り 100 件が維持されたことは regression gate として残す。

次の分割対策では v2 の lexical cost を固定した上で、manual overlay の一文字 fragment を機能語用法と内容語内部用法で分ける calibration、または既報の小重み `matrix.def` hybrid を held-out 条件で評価する。

旧 `dictionary_learned.tsv` は上書きせず、バグ影響下の historical baseline として保持する。

`src/` を含む production コードは変更していない。

Git commit / push は行っていない。
