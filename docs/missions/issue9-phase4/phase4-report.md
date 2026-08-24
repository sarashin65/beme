# Issue #9 Phase 4 Report

## 1. Scope and Reproducibility

Phase 4 は、Top-1 定義の統一、K-best と文脈リランカを分離評価するハーネス、集計・統計 API、d1 を含むリーク監査、局所モデル単独（L）/ リランカ単独（R）/ 併用（LR）の実測、cascade gate の可否判定を対象にした。結論は PLAN §5 の **「効いた」分岐**である。d1 test の contrast pair 128 件で LR は L に対して改善方向かつ McNemar `p<0.05` で、seed 9/17/33 の全 seed が同方向・有意だった。

実装・テストは miniPC、CE score とリーク監査は `beme-gpu` の `/home/sarashin/beme-p4` で実行した。両環境の commit は `d1bf01b99c5d3bbcf47ce36aff97f6e6c641d875`、branch は `feat/issue9-phase4` である。`beme-gpu` は 24 core を CE score に使い、S2/S3/S4 は各 `--torch-threads 8` の 3 process で並列実行した。IME レイテンシは Intel N100・4 core の miniPC（host `WIN-I7DHB6T7L79`）でのみ測った。GPU 側の score elapsed と miniPC の IME レイテンシは混ぜない。

入力 SHA-256:

| input | SHA-256 |
|---|---|
| `data/d1/train.jsonl` | `b9609547a66d8b15ab72d416b1f964b62881fa1d77d3ddd2cd900d7d3c23baa8` |
| `data/d1/dev.jsonl` | `8a6aea13dcc1caf61708cf94f9db836e14ba3544fa5c4c522d9668d418182f47` |
| `data/d1/test.jsonl` | `7b0ad8ff3f9abf11ccfb9488214db0199ecd9b06d07928dea0fb6c8681f6f2ff` |
| `data/dictionary_v2.tsv` | `1e423c314cb6ee117cb429cb7a9a063a199f4c3e80d9f95ffcb584f855566e0a` |
| A0 seed 9 weights | `70a6ffed5405bbc4747e83eb36bc0a7213800263ce1db5e2c47d8dd438145fee` |
| A0 seed 17 weights | `c53101d2b0eef77981b0e1063078b3558ada34c1a4265c8fd80dbb2298a2bc6a` |
| A0 seed 33 weights | `b3545fea89f5eda7881e35f075d6245ad0faa83084db1f03ed51c56a779d07b0` |

Cross-encoder は `hotchpotch/japanese-reranker-base-v2`、Hugging Face snapshot revision は `f21b54987ac864d7a355f3384bf05d0939e31cb1` である。`model.safetensors` の SHA-256 は `f269ef08964c2b6b0ad78f972650d645f4d00232b2da7ceda02d2fa050a73a29`。全 model file は `artifacts/issue9-phase4/S1/model_files.sha256`（artifact 自体の SHA-256 `83e00581f3226d423bb6faf5ae09119e34f5e518b2fc5bbbdde304983981ab95`）に記録した。

主な実行コマンドを逐語で示す。Task 1–3 の実装検証:

```bash
uv run pytest -q tests/test_top1_unification.py
uv run pytest -q tests/test_rerank_eval.py
uv run pytest -q tests/test_rerank_stats.py
uv run pytest -q
```

リーク監査:

```bash
/home/sarashin/.local/bin/uv run python scripts/audit_leakage.py \
  --d1-dir data/d1 \
  --ajimee-cases data/ajimee_bench_jwtd_v2_v1.json \
  --mozc-cases data/mozc_oss.tsv \
  --cases data/evaluation_cases.jsonl \
  --output artifacts/issue9-phase4/audit/leakage.json --json
```

CE score。S1 だけ model を取得し、以後は local files only とした。

```bash
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py score --data data/d1/dev.jsonl --dictionary data/dictionary_v2.tsv --weights /home/sarashin/beme-p3/artifacts/issue9-phase3/A0/seed9/weights.json --reranker cross-encoder --torch-threads 8 --output artifacts/issue9-phase4/S1/scores.jsonl
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py score --data data/d1/test.jsonl --dictionary data/dictionary_v2.tsv --weights /home/sarashin/beme-p3/artifacts/issue9-phase3/A0/seed9/weights.json --reranker cross-encoder --cross-encoder-local-files-only --torch-threads 8 --output artifacts/issue9-phase4/S2/scores.jsonl
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py score --data data/d1/test.jsonl --dictionary data/dictionary_v2.tsv --weights /home/sarashin/beme-p3/artifacts/issue9-phase3/A0/seed17/weights.json --reranker cross-encoder --cross-encoder-local-files-only --torch-threads 8 --output artifacts/issue9-phase4/S3/scores.jsonl
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py score --data data/d1/test.jsonl --dictionary data/dictionary_v2.tsv --weights /home/sarashin/beme-p3/artifacts/issue9-phase3/A0/seed33/weights.json --reranker cross-encoder --cross-encoder-local-files-only --torch-threads 8 --output artifacts/issue9-phase4/S4/scores.jsonl
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py score --data data/d1/test.jsonl --dictionary data/dictionary_v2.tsv --weights /home/sarashin/beme-p3/artifacts/issue9-phase3/A0/seed9/weights.json --reranker mock --output artifacts/issue9-phase4/S5/scores.jsonl
```

dev 選択と test 集計:

```bash
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S1/scores.jsonl --viterbi-weight 0,0.25,0.5,1,2,3.94,8 --rerank-k 5,10,20,64 --output artifacts/issue9-phase4/C1/grid.json --json
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S2/scores.jsonl --viterbi-weight 8 --rerank-k 10 --gate=-1 --output artifacts/issue9-phase4/C2/L.json
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S2/scores.jsonl --viterbi-weight 0 --rerank-k 64 --output artifacts/issue9-phase4/C2/R.json
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S2/scores.jsonl --viterbi-weight 8 --rerank-k 10 --output artifacts/issue9-phase4/C2/LR.json
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S3/scores.jsonl --viterbi-weight 8 --rerank-k 10 --output artifacts/issue9-phase4/C3/seed17.json
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S4/scores.jsonl --viterbi-weight 8 --rerank-k 10 --output artifacts/issue9-phase4/C3/seed33.json
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S5/scores.jsonl --viterbi-weight 8 --rerank-k 10 --output artifacts/issue9-phase4/C4/mock.json
/home/sarashin/.local/bin/uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S2/scores.jsonl --viterbi-weight 8 --rerank-k 10 --gate=-1,0,0.25,0.5,1,1.5,2,2.025162545128425,2.5,3,4,5,8,1000000 --output artifacts/issue9-phase4/C5/curve_raw.json
```

回帰と Task 6:

```bash
uv run python scripts/evaluate.py --suite all --reranker mock --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json --json
uv run python scripts/rerank_eval.py combine --scores artifacts/issue9-phase4/S2/scores.jsonl --viterbi-weight 8 --rerank-k 64 --output artifacts/issue9-phase4/C6/full_candidates_test.json
uv run python /tmp/task6_latency.py
uv run pytest -q
```

`/tmp/task6_latency.py` は Task 6 の一時測定 driver で、辞書・A0 seed 9・同じ CE revision をロードし、d1 test 先頭50件を warm-up 1件除外後に `rerank_k=10` と全候補で測った。driver は測定後に削除したため、コマンドだけからの再実行性は残っていない。この制約を推測で補わず、入力 hash・条件・出力を `artifacts/issue9-phase4/C6/latency_minipc.json` に保存した。

主要 artifact と SHA-256:

| artifact | SHA-256 |
|---|---|
| `artifacts/issue9-phase4/selection_rule.json` | `68446e79907c81fb4d3a10d1316133724b25686673f4a54366308d4fcff642f6` |
| `artifacts/issue9-phase4/audit/leakage.json` | `da1941b75031c4fea92587a6950418f273adbc7c01250a293ed4bc8ae3be207d` |
| `artifacts/issue9-phase4/C1/grid.json` | `472342f5dd1d257a75131387e2a794c1b0e86cb8bc39bb05145553e9a9716bbb` |
| `artifacts/issue9-phase4/C1/selected.json` | `9ed603b91ca0d227d50d5c55edcf1fb984ef45151b9c951982f0611377269efb` |
| `artifacts/issue9-phase4/C2/L.json` | `970384bb5e8e6b4dd9bf4b263e2ec389aed0426eedb64088cab2bde79c68fa7b` |
| `artifacts/issue9-phase4/C2/R.json` | `da895fad2b5e361a037b838b2a3d27ecd03783774043078be556aa2f8fcc2010` |
| `artifacts/issue9-phase4/C2/LR.json` | `0fffcfd3125460294ad47ecd40459034b8dbef37dc372415ffccf8ba7fc5dd31` |
| `artifacts/issue9-phase4/C2/lost_cases.json` | `bd650a1e6201747674538e5a4e0301d7f5f73467fb5d8ed075e2e6454380f662` |
| `artifacts/issue9-phase4/C2/contrast_lost_cases.json` | `e3f912ec9897f1761556d57c6c90701fb98c4382a7556bb1f5d1c0885cb65378` |
| `artifacts/issue9-phase4/C2/stratification.json` | `1fe85b75fd247eb494ed37fd512a8a2af25214e534bd608f5f0721d42a968c3a` |
| `artifacts/issue9-phase4/C2/surface_recall_invariance.json` | `7454852528b21c5c56842e54d6894aab9f39fa44e782b8a9ec7ce03decda0d70` |
| `artifacts/issue9-phase4/C3/mcnemar_by_seed.json` | `0e19a7590c6d4f7000593da655bfc33e37b497479dbe09769adf7274ba840416` |
| `artifacts/issue9-phase4/C3/seed17.json` | `7a4f2676f790c567840765b102bca7b5066cc723e54265cae2f3b2dc82a8df54` |
| `artifacts/issue9-phase4/C3/seed33.json` | `8356201fd705fd33a5cbe5da6082af3670729544ea5cc94d61050842f8b1344f` |
| `artifacts/issue9-phase4/C4/mock.json` | `71c23195791e31629360f468f1c1e533acfb78fcd8cfe9a8666ab47b5e61c0dc` |
| `artifacts/issue9-phase4/C5/curve.json` | `034b493e52964f6962fb1dba4fa33aab9d3b12bef203882af0ad2b451ddcb443` |
| `artifacts/issue9-phase4/C6/decision.json` | `4ac3d94f4af38a73c7dedad4125e6d81c477186b4d827cfc382de28bbbf6e2f2` |
| `artifacts/issue9-phase4/C6/gate_dev.json` | `329f022c3c6e1116044a36d485861300c261572c3e67d9ecf7ffe340a2c7d73b` |
| `artifacts/issue9-phase4/C6/full_candidates_test.json` | `3bd6bd2c0fc6261406e0427ea58b483d73718c3a417eb184b27bf1fe191ff4e9` |
| `artifacts/issue9-phase4/C6/latency_minipc.json` | `cf89c08ecde55419c1779de2065de285cf144a581f7826b3bda1cb83a7080e36` |
| `artifacts/issue9-phase4/C6/invariance_suite_all_mock.json` | `ce51a635a93bb24692fd10ea5f4cfa8e18135a2ef52593dce34048d75d652048` |
| `artifacts/issue9-phase4/latency_minipc/summary.json` | `70e819cd1b9a8926aadac53c9f34a009b7f832daa09cabe9081f9e33b3e92aa9` |
| `artifacts/issue9-phase4/regression/evaluate_all_mock.json` | `de72da6ec4d0788c012dcba81b4267011e894cbd74f7ce2d1006f9925cdaede1` |

`artifacts/` は現時点で git 管理外かつ `.gitignore` 対象外である。レポートを後から再検証するには、ユーザーがこれらの保存方法を決める必要がある。

## 2. Top-1 Definition Unification

Phase 3 report §10.3 は、A3 の `end_to_end_errors=588` と converter 経由の誤り `796` に 208 件の乖離があり、原因が「同一最小 cost のとき `k_best_viterbi` は辺列挙順、Converter は `(cost, surface)` 順」という Top-1 定義差だと特定した。Phase 4 Task 1 は `top1_surface()` を共有入口として `evaluate_split` を Converter と同じ `(cost, surface)` 最小へ統一した。

Task 1 の実測では A3 の 208 件の乖離は解消し、同点が発生しない残り11条件の結果は不変だった。回帰テストは、同 cost、9本超の同点、非接続 lattice、同点なし、`evaluate_split` と実 Converter の一致を固定する。`k_best_viterbi` の A* heap key 自体は変更していない。

この結果は「A3 の性能が改善した」という意味ではない。学習時評価と成果物評価が同じ Top-1 を数えるようになった、という測定定義の修正である。

## 3. Separating Oracle Recall from Top-1

チェックリスト「oracle recall と Top-1 改善を分けて評価」への結果である。選択点 `rerank_k=10` について、候補集合上限、L から残る伸びしろ、LR の実績を分ける。

| split/scope | 候補上限 `surface_recall@10` | L Top-1 | 上限までの伸びしろ | LR Top-1 | 実績 `LR-L` |
|---|---:|---:|---:|---:|---:|
| d1 test 全体 2,000 | .988000 | .731000 | .257000 | .831500 | +.100500 |
| d1 test contrast 128 | .960938 | .328125 | .632813 | .531250 | +.203125 |

`surface_recall@64` は全体・contrast とも `1.0` だった。上限が高いことと、Top-1 が実際に改善することは別である。LR は全体で255件、contrastで33件を救った一方、全体54件、contrast7件を失ったため、上限の全てを実現したわけではない。

`combine` は各例について rerank 前後の先頭 `min(rerank_k, candidate_count)` の surface 集合が同一かアサートする。C1 28点が完走し、C2 の明示検査でも2,000件の違反は0だった。

| cutoff | before | after | violations |
|---:|---:|---:|---:|
| 10 | .988000 | .988000 | 0/2,000 |
| 64 | 1.000000 | 1.000000 | 0/2,000 |

正本は `artifacts/issue9-phase4/C2/surface_recall_invariance.json` である。

## 4. Grid Selection on dev

選択規則は score 実行前に `artifacts/issue9-phase4/selection_rule.json` へ固定した。

```text
selection split: d1 dev 1,000 examples
primary: overall Top-1 accuracy
secondary: contrast pair Top-1 accuracy, 44 examples
tie break: overall desc, contrast desc, rerank_k asc, viterbi_weight desc
seed policy: seed 9 dev だけで選択し、seed 17/33 へ同じ点を適用
```

train contrast 872件は局所モデルの学習対象であり、局所 cost が楽観側に偏るため選択に使っていない。test は選択後に評価した。

| viterbi weight | k=5 overall/contrast | k=10 | k=20 | k=64 |
|---:|---:|---:|---:|---:|
| 0 | .716/.545455 | .662/.522727 | .602/.545455 | .540/.500000 |
| .25 | .772/.545455 | .760/.568182 | .740/.590909 | .713/.568182 |
| .5 | .796/.568182 | .792/.590909 | .786/.613636 | .775/.590909 |
| 1 | .813/.568182 | .810/.590909 | .808/.613636 | .806/.613636 |
| 2 | .835/.568182 | .832/.590909 | .832/.613636 | .833/.613636 |
| 3.94 | .844/.568182 | .843/.613636 | .842/.590909 | .842/.590909 |
| 8 | .852/.590909 | **.854/.613636** | .853/.590909 | .853/.590909 |

選択点は `viterbi_weight=8`, `rerank_k=10`。`vw=8` の k 方向は `.852/.854/.853/.853` で高原上にあり、単独スパイクではない。一方、`vw=8` は事前登録グリッドの上端なので、weight 方向の高原を両側から bracket できていない。より大きい weight が良いかは判別できない。

## 5. Local / Reranker-only / Combined

チェックリスト「局所モデル単独・リランカ単独・併用を比較」への結果である。条件を次のように定義した。

- **L（局所モデル単独）**: A0 局所 cost 順。リランカを呼ばない。
- **R（リランカ単独）**: 候補生成は L と同一。IME は候補生成を省けないため「候補生成なし」ではない。順位付けだけを CE relevance のみにし、`viterbi_weight=0`, `rerank_k=64` とした。
- **LR（併用）**: 同じ候補生成に dev 選択点 `viterbi_weight=8`, `rerank_k=10` を適用した。

primary は d1 test contrast 128件、secondary は全体2,000件である。contrast を primary にしたのは、全体には文脈で順位を変える必要の薄い例が多く、文脈識別力を直接測る contrast pair のほうがリランカ効果に感度があるためである。

| condition | overall Top-1 | contrast Top-1 | pair: both / one / wrong | overall `b/c/p` | contrast `b/c/p` |
|---|---:|---:|---:|---:|---:|
| L | .731000 | .328125 | 0 / 42 / 22 | 0/0/1.0 | 0/0/1.0 |
| R | .521000 | .421875 | 14 / 26 / 24 | 639/219/`2.0533e-46` | 18/30/`.112351` |
| LR | **.831500** | **.531250** | 21 / 26 / 17 | 54/255/`5.4066e-30` | 7/33/`7.7227e-05` |

`b` は L で正解・比較条件で不正解になった loss、`c` は gain である。R の overall は有意に悪化した。R の contrast は改善方向だが、128件では `p=.112351` で有意改善とは判別できない。LR は overall と contrast の両方で改善方向・有意だった。

pair は64対すべて完全で `incomplete_pairs=0`。LR は「両方正解」を0対から21対へ増やしたが、「片方だけ正解」は42対から26対になった。これは片側だけでなく対の両側を正解へ動かした例があることを示す。

mock 対照は overall/contrast とも `b=c=0`, `p=1.0` で L と完全一致した。集計経路自体が改善を作っていないことを確認した。

LR の層別 error rate:

| layer | n | before | after | note |
|---|---:|---:|---:|---|
| contrast=false | 1,872 | .241453 | .147970 | |
| contrast=true | 128 | .671875 | .468750 | primary |
| proper noun=false | 1,756 | .268793 | .162301 | |
| proper noun=true | 244 | .270492 | .213115 | |
| reading length 1–3 | 637 | .350078 | .227630 | |
| reading length 4–6 | 1,187 | .235889 | .138163 | |
| reading length 7–10 | 153 | .202614 | .169935 | |
| reading length 11–16 | 23 | .173913 | .086957 | low support |
| span_tokens=1 | 1,730 | .269364 | .161850 | proxy |
| span_tokens=2 | 143 | .223776 | .174825 | proxy |
| span_tokens=3 | 127 | .314961 | .251969 | proxy |

reading length 11–16 は `n=23` なので改善を一般化できるか判別できない。`span_tokens` は変換品質そのものではなく proxy である。正本は `artifacts/issue9-phase4/C2/stratification.json`（SHA-256 `1fe85b75fd247eb494ed37fd512a8a2af25214e534bd608f5f0721d42a968c3a`）。

## 6. Seed Stability

単一 seed の p 値で有意ラベルを付けていない。`rerank_stats.label` は複数 seed の各分割表を個別に検定し、全 seed が `p<.05` かつ同方向の場合だけ有意とする。seed 間で分割表を合算する API はない。

| seed | overall before → LR | `b/c/p` | contrast before → LR | `b/c/p` |
|---:|---:|---:|---:|---:|
| 9 | .7310 → .8315 | 54/255/`5.4066e-30` | .328125 → .531250 | 7/33/`7.7227e-05` |
| 17 | .7365 → .8320 | 47/238/`2.1975e-29` | .328125 → .531250 | 5/31/`3.0909e-05` |
| 33 | .7310 → .8310 | 47/247/`3.8455e-31` | .328125 → .531250 | 5/31/`3.0909e-05` |

実出力は overall/contrast とも `significant_better`。従って PLAN §5 の「効いた」分岐と判定した。

Task 3 後に実装者と別モデルで集計・検定部分をクロスレビューした。`paired_table` の母集団固定、単一 seed と pooled table の拒否、全 seed 規則、Converter と score sort の一致、pair/histogram 契約を確認し、重大な不具合は見つからなかった。軽微な残リスクとして、`stratify(..., "span_tokens")` は観測 bucket だけを返すため、将来の利用側が固定 bucket を仮定すると `KeyError` になり得るとの指摘があった。現契約違反ではないため Phase 4 では変更していない。

## 7. What the Reranker Broke

LR seed 9 は全体255件を救い、54件を失った。以下は loss `b=54` の全件である。`contrast` は contrast pair 由来、分類は結果を読みやすくするための事後的な手作業分類で、意味的同値性の gold 判定ではない。表記揺れでも exact-match 指標では loss である。

| id | reading | gold | LR Top-1 | class | contrast |
|---|---|---|---|---|---|
| `00d74685d75f75cba3e1` | おもっ | 思っ | 重っ | semantic/homophone | |
| `01d8f21952012a8bf281` | のぞむ | 臨む | 望む | semantic/homophone | yes |
| `0764f5e6c513aa140a54` | こうぎょう | 工業 | 鉱業 | semantic/homophone | yes |
| `0b4b1a4b3996d2b6a60c` | てきした | 適した | 敵した | semantic/homophone | |
| `0bd4e64752db32754e7f` | うたっ | 歌っ | 唄っ | orthographic/lexical variant | |
| `14dfe4f9192822465ab7` | せんそう | 戦争 | 船倉 | semantic/homophone | |
| `1555be9e76a1271685e8` | ならん | 並ん | 列ん | semantic/homophone | |
| `17e806d3bc09fb313f05` | いこう | 以降 | 移行 | semantic/homophone | |
| `1bbdaba316bda320017f` | きかん | 期間 | 機関 | semantic/homophone | yes |
| `2587b7d022639ed948c5` | せいたい | 生態 | 整体 | semantic/homophone | yes |
| `295be04d14360880d095` | いけにえ | 生贄 | 生け贄 | orthographic/lexical variant | |
| `2c903b034cbdeda462b2` | あまがさ | 雨傘 | 雨がさ | orthographic/lexical variant | |
| `2e9103ac5b83a8f892c1` | いっしょ | 一緒 | 一所 | semantic/homophone | |
| `2ebd2a735902035e88d2` | れんさいかいし | 連載開始 | 連載怪死 | semantic/homophone | |
| `32d3960541fe7a219935` | たいけい | 体系 | 大兄 | semantic/homophone | |
| `35a2aa9a67b5aeb4fc6d` | さいこう | 最高 | 再興 | semantic/homophone | |
| `3b5df0f274af5d105390` | かくさん | 拡散 | 核酸 | semantic/homophone | |
| `3ca9bd695639f9245782` | みたさ | 満たさ | 充たさ | orthographic/lexical variant | |
| `3df8cdbbd948b7928aba` | しょき | 書記 | 所期 | semantic/homophone | yes |
| `429724af14cbdf3c0e2c` | はいき | 廃棄 | 排気 | semantic/homophone | |
| `47d7a4a44fad50c189d8` | じゅんじいんたいし | 順次引退し | 順次隠退し | semantic/lexical choice | |
| `4982a133159136e84073` | とびこし | 飛越 | 飛越し | inflection/segmentation | |
| `4af1f62066221b5e10ed` | いし | 医師 | 良し | semantic/homophone | yes |
| `4be4249002c528a08e7a` | じょうほう | 情報 | 定法 | semantic/homophone | |
| `4c813720e0c2a27c0901` | しぜんかがく | 自然科学 | 自然化学 | semantic/homophone | |
| `4d822c99a0624b6d7bcd` | まじり | 混じり | 交じり | orthographic/lexical variant | |
| `4e7518112390ba4b6b90` | のこうしきせかい | の公式世界 | の硬式世界 | semantic/homophone | |
| `50650b242651cd960ab9` | さいご | 最期 | 最後 | semantic/lexical choice | yes |
| `55360de37e6e8d9c72aa` | ひていは | 否定は | 比定は | semantic/homophone | |
| `5536f0fb6a3d07d3766e` | しゅんいち | 俊一 | 駿一 | personal-name ambiguity | |
| `58f26e3c47033651b74d` | せんげん | 宣言 | 選言 | semantic/homophone | |
| `61bd05f63860ce169de1` | さいじ | 催事 | 祭事 | semantic/homophone | |
| `67bd449f5babeeb78d48` | しょうにん | 承認 | 証人 | semantic/homophone | |
| `6a00a972a3a0728f52a1` | きょうし | 教師 | 狂詩 | semantic/homophone | |
| `786391ad609a7ae87d12` | ちゅうじょう | 中将 | 衷情 | semantic/homophone | |
| `78b681a48aa20411e9b0` | しげん | 資源 | 始原 | semantic/homophone | |
| `7bae723a93466e0512c4` | じんせいにたいして | 人生に対して | 人性に対して | semantic/homophone | |
| `7f08b803ae492b76ab97` | ちいき | 地域 | 値域 | semantic/homophone | |
| `8d63e51a4f779e95c706` | できあがり | 出来上がり | 出来上り | orthographic/lexical variant | |
| `8df0b50a51e57919fef9` | ほてる | ホテル | 熱る | semantic/homophone | |
| `9042ddd6ad1ad6911122` | えいさく | 栄作 | 英作 | personal-name ambiguity | |
| `91496dd26d9443907321` | ちょうせい | 調整 | 調製 | semantic/homophone | |
| `919e4f263b5ee418179e` | ちょうしゅう | 徴収 | 聴衆 | semantic/homophone | |
| `94c8f4dc5075b8601ae0` | すべりこま | 滑り込ま | 辷り込ま | orthographic/lexical variant | |
| `a43cae48d910a4ac9630` | せいすう | 整数 | 正数 | semantic/homophone | |
| `a47e14b377be1fb2eb68` | まんがの | マンガの | 漫画の | orthographic/lexical variant | |
| `a4e0cd05cc9b7c383c39` | せんせん | 戦線 | 宣戦 | semantic/homophone | |
| `bad11fbae928cf0d0458` | しゃない | 車内 | 社内 | semantic/homophone | |
| `cf8cbb596ca47f33c945` | かんれん | 関連 | 関聯 | orthographic/lexical variant | |
| `d22efb407a2568f1ae62` | ちかづい | 近づい | 近付い | orthographic/lexical variant | |
| `db31468e4f946cfec8c0` | かまがや | 鎌ケ谷 | 鎌ヶ谷 | orthographic/place-name variant | |
| `dd3cfe06dbd2f0573fd6` | ぞくへん | 続編 | 続篇 | orthographic/lexical variant | |
| `f77b9a8a3a0e72ef6ec6` | きよひと | 清仁 | 清人 | personal-name ambiguity | |
| `fd7e59c65db980fb4b9c` | なかっ | 無かっ | なかっ | orthographic/lexical variant | |

contrast loss は表中の7件。正本の subset は `artifacts/issue9-phase4/C2/contrast_lost_cases.json`（SHA-256 `e3f912ec9897f1761556d57c6c90701fb98c4382a7556bb1f5d1c0885cb65378`）。分類から、CE が文脈に基づく同音異義語選択を誤った例だけでなく、表記揺れ・人名候補を exact gold から動かす例もあることが分かる。表記揺れが利用者に許容されるかは本評価から判別できない。

## 8. gold_rank Distribution

候補集合の `absent=0` なので、「正解が候補に無い」のではなく順位付けが課題だった。読みやすさのため rank 6–10 と11–20を集約する。完全な1–20分布は C2各 artifact を正本とする。

### 全体 2,000 件

| order | rank1 | rank2 | rank3 | rank4 | rank5 | rank6–10 | rank11–20 | >20 | absent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L / before | 1462 | 257 | 117 | 52 | 29 | 59 | 21 | 3 | 0 |
| R after | 1042 | 239 | 127 | 64 | 70 | 192 | 139 | 127 | 0 |
| LR after | 1663 | 196 | 57 | 31 | 12 | 17 | 21 | 3 | 0 |

### contrast 128 件

| order | rank1 | rank2 | rank3 | rank4 | rank5 | rank6–10 | rank11–20 | >20 | absent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L / before | 42 | 36 | 19 | 9 | 4 | 13 | 5 | 0 | 0 |
| R after | 54 | 11 | 16 | 4 | 3 | 14 | 11 | 15 | 0 |
| LR after | 68 | 31 | 9 | 6 | 4 | 5 | 5 | 0 | 0 |

R は relevance だけで深い候補を上げ、全体127件、contrast15件を rank `>20` へ押し下げた。LR は局所 cost を守りとして残し、rank1を増やしつつ `>20` を増やさなかった。これは「局所 cost と relevance の結合が効果の本体」という3条件比較と整合する。

## 9. Latency

以下はすべて Intel N100・4 core miniPC、host `WIN-I7DHB6T7L79` の実測である。モデルロードと1件の warm-up は除外した。GPU machine `sarashin` の score elapsed は実験完走時間であり、この表へ混ぜていない。

d1 test 先頭50件、同じ CE revision での実測:

| setting | scored candidates | mean / conversion | median | p95 | mean / scored candidate | first-50 Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| 全候補、gateなし | 2,308 | 3.321963 s | 3.197742 s | 6.316915 s | 71.747 ms | .72 |
| `rerank_k=10`、gateなし | 500 | .739371 s | .689987 s | 1.159115 s | 73.454 ms | .72 |

miniPC `WIN-I7DHB6T7L79` で `rerank_k=10` は平均変換時間を77.74%削減した。d1 test 2,000件の contrast Top-1 も、全候補 LR `.523438` から選択した k=10 LR `.531250` へ低下しなかった。ただしこの `.007812` 差に統計検定は行っていないため、k=10のほうが精度も高いとは判別しない。

Task 5 の別実走行も同じ miniPC `WIN-I7DHB6T7L79` で、全候補相当50件の平均 `3.393213 s/変換`、`73.510 ms/候補` だった。この値からの k別値は候補数比例の投影であり、上表の Task 6 実測とは区別する。正本は `artifacts/issue9-phase4/latency_minipc/summary.json`（SHA-256 `70e819cd1b9a8926aadac53c9f34a009b7f832daa09cabe9081f9e33b3e92aa9`）と `artifacts/issue9-phase4/C6/latency_minipc.json` である。

参考として GPU machine `sarashin`・24 core で記録された score elapsed の例平均は S1 `.5744 s`、並列実行したS2/S3/S4がそれぞれ `1.5758/1.5854/1.5851 s`、mock S5 `.00246 s` だった。S2/S3/S4 は互いに資源を共有した並列 process なので、これらを miniPC の単発IMEレイテンシと比較してはならない。

## 10. Cascade Gate

Task 6 のゲート評価は実施したが、**配線は実施しなかった**。理由は PLAN §5 の「効いた」分岐に進んだ後、Task 6 の独立した停止条件「`recall≥.95` を満たす最小 fire rate が `.5` を超えるなら配線しない」に該当したためである。

dev 1,000件の LR rescue 117件を分母にした結果:

| candidate | condition | rescue recall | fire rate | decision |
|---|---|---:|---:|---|
| confidence 再較正出発点 | `<2.0251625451` | 1.000 | .702 | reject |
| confidence C5候補 | `<.25` | 1.000 | **.567** | reject: `.5`超 |
| confidence + 上位2表層長一致 | threshold 全域 | 最大 .837607 | — | reject: recall不足 |
| top-5 cost span | `<4` | 1.000 | .899 | reject |

`m=5` は dev rescue の rerank前 gold rankを96.58%覆う最小値として決めた。devで選んだ最良の登録済み候補 `confidence<.25` を既存C5でtest 1回確認すると、recall `1.0`, fire rate `.576`, overall Top-1 `.8315`, contrast Top-1 `.53125` だった。

未登録の観測値に事後適合した dev threshold `<.1299747984` は recall `.957265`, fire `.464` だったが、Task 5 C5 で固定されていないため採用しなかった。これを Phase 4 の成功設定として報告すると選択規律を破る。

`confidence is None` は現行契約どおり `needs_rerank=False`、すなわち発火しない。配線を入れなかったため `ConversionResult.needs_rerank/reranked` と `to_dict()` の意味は変更していない。`converter.py`、`evaluate.py`、`rerank_eval.py` に Task 6 の tracked diff はない。

反実仮想として、miniPC `WIN-I7DHB6T7L79` の k=10実測とtest fire rate `.576` を組み合わせると期待値は約 `.427929 s/変換` になる。この値は実配線の実測ではなく、測定済みCE時間と発火率からの算出値である。目標レイテンシは設定しておらず、許容可否はユーザー判断とする。

## 11. Leakage Audit

Phase 3 report §8 では d1 JSONL が miniPC に無く、document n-gram、reading、vocabulary overlap、split再計算が `unverifiable` だった。Phase 4 は d1 を持つ GPU machine `sarashin` の新規 clone で監査を完走した。

| item | result |
|---|---|
| manifest consistency | `all_passed=true`、13/13 checks passed |
| split recomputation | `verified=true`, mismatches `[]` |
| failures | `[]` |
| unverifiable | `[]` |
| AJIMEE document leakage | matched 0/5,777 n-grams, `verified=true` |
| handmade document leakage | matched 0/4 n-grams, `verified=true` |
| Mozc source-document claim | `verified=true`、原文/context fieldなし |

Mozc は assertion 500件のうち評価対象となる一意 reading で30/465、vocabularyで9/481がd1 trainと共有された。しかし reading/vocabulary sharing は監査定義上 document leakage ではない。Mozc の保持fieldは `id`, `label`, `reading`, `expected`, `command`, `expected_rank` で、原文・contextは無い。したがって Phase 3 の「読みと assertion のみで原文を持たず、語彙共有を文書リークとみなさない」という主張を d1 ありで追検証できた。

ここから一般的な「リークが一切ない」とは結論しない。監査が検証した field と n-gram 定義の範囲で document matchが無く、manifest/splitが整合した、という結論である。

## 12. Negative and Null Results

- R（候補生成は同一、順位だけ relevance）は overall を `.731→.521` へ有意に悪化させた。contrast は `.328→.422` の改善方向だが `n=128`, `p=.112351` で有意改善とは判別できない。
- mock 対照は `b=c=0`, `p=1.0`。集計・ハーネスだけによる変化はなかった。
- confidence 単独 gate は dev fire rate `.567`、test `.576`。停止基準 `.5` を満たさず、cascade配線を実施しなかった。
- confidenceと上位2表層長一致の積は最大recall `.837607` で、必要な`.95`に届かなかった。top-5 cost spanはrecallを満たすとfire `.899`だった。
- devの選択点はk方向の高原上だが `viterbi_weight=8` がグリッド上端で、weight方向の最適域は判別できない。
- dev contrast は44件しかないため副指標とし、全体1,000件を主指標にした。44件だけから設定の安定性は判別できない。
- reading length 11–16 の層は `n=23`。error rateが下がったが、この標本サイズから一般化できるか判別できない。
- context empty のtest 18件はrelevanceを`null`としてリランクしていない。
- 既存 handmade/AJIMEE/Mozc suite は回帰確認だけに使った。handmadeはmockで既にTop-1 1.0、Mozcはcontextを持たずCEが走らず、AJIMEEは文分割が混ざるため、LR採否の根拠にはしていない。
- feature spec v2、局所モデル再学習、matrix/frequency条件X0での再測定、D1小型LM、LLM logits再評価は実施しなかった。Phase 4 はA0とCE接続の効果を分離するscopeだったためである。
- PLAN §2.4 の40件パイロットは結果・分岐判定の根拠に使用していない。

## 13. Limitations and Handoff

### Issue #9 チェックリストへの回答

| checklist | result |
|---|---|
| 局所モデルの K-best を #2/#5 のリランカへ渡す | 完了。score/combineの2段分離でCE relevanceを1回だけ計算し、同じK-best候補へ適用した。#5 D1小型LMは既存No-GoのためCEを実測した。 |
| oracle recall と Top-1 改善を分けて評価 | 完了。`surface_recall@10=.988`（全体）/`.960938`（contrast）を不変アサーションで固定し、LR実績をそれぞれ`+.1005`/`+.203125`として別に示した。 |
| 局所モデル単独・リランカ単独・併用を比較 | 完了。L `.731/.328`、R `.521/.422`、LR `.8315/.531`。LRは3 seed全てで有意改善方向だった。 |

### 設計上の残る制約

1. CE + `rerank_k=10` は miniPC `WIN-I7DHB6T7L79` で `.739 s/変換`。全候補 `3.322 s` より短いが、IMEとして許容できる目標値は定義していない。
2. 安全なcascade gateは見つからず配線していない。未登録thresholdの追試をする場合は、新しいdev/testまたはnested splitで規則を先に固定する必要がある。
3. `viterbi_weight=8` はdev grid上端。より大きいweightを探索する場合もtestを再選択に使わない。
4. loss 54件には同音異義語だけでなく表記揺れ・人名がある。exact-match以外の許容表記評価とcase分類が必要である。
5. artifactは現在git管理外である。レポートの再現性を維持する保存先を決める必要がある。

### 起票する別 issue（起票自体はユーザー判断）

1. **学習済み重みの配布形態**: Phase 1から3フェーズ持ち越し。`pyproject.toml` にpackage-data指定がなく、wheel installで`default_scorer()`が動かない。学習済みweightは外部path必須、同梱対象は`legacy_v1.json`と`decomposed_v1.json`に限定する方針を具体化する。
2. **評価セット拡張**: Phase 3 report §9–10 はd1 test 2,000件では局所モデル改善側を判別できない条件があった。Phase 4のprimary 128件ではLRを判別できたが、dev contrastは44件で、層別の一部は23件しかない。contrast・人名・表記揺れを拡張する。
3. **X0（matrix + frequency）でのリランカ再測定**: PLAN §5「効いた」分岐に従い、局所モデル側の候補cost変更とCE結合の相互作用を独立dev/testで測る。
4. **AJIMEEへの文脈供給**: 現行AJIMEE評価は文分割とcontext条件がd1と異なる。CEの文脈効果を測れる入力設計を別issueで扱う。
5. **#2 頻度バイアス / #3 誤り分類への引き継ぎ**: Rが深い候補を上げる問題、LR loss 54件の同音異義語・人名・表記揺れをcase-levelで扱う。

テスト数は Phase 3 完了時の585件から613件へ増えた。最終 `uv run pytest -q` は `613 passed`。git commit、git push、issue投稿はこのタスクでは行わない。
