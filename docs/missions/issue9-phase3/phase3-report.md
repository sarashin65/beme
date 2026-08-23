# Issue #9 Phase 3 Report

## 1. Scope and Reproducibility

Phase 3 は、feature-spec 切替による 12 条件の ablation、d1 の学習・評価、既存評価 suite、oracle recall、層別評価、case-level 診断、cost-scale 測定、leakage audit を対象にした。

学習・d1 評価は `beme-gpu` の `~/beme-p3` で実行した。作業ツリーの commit は `194131dadc50bb8ea989c17d56fbed0e2b1539f5` と記録されている。d1 split は GPU 側 `~/beme-d1/repo/data/d1/` から symlink で配置した。`data/d1/*.jsonl` の SHA-256 は manifest に記録された train=`b9609547a66d8b15ab72d416b1f964b62881fa1d77d3ddd2cd900d7d3c23baa`、dev=`8a6aea13dcc1caf61708cf94f9db836e14ba3544fa5c4c522d9668d418182f47`、test=`7b0ad8ff3f9abf11ccfb9488214db0199ecd9b06d07928dea0fb6c8681f6f2ff8` である。

記録された入力 SHA-256 は dictionary_v2=`1e423c314cb6ee117cb429cb7a9a063a199f4c3e80d9f95ffcb584f855566e0a`、初期 `decomposed_v1` weights=`029e82a617ed3e339a55be2cb85da486fb5777ac51442cb0a8d4faf0ff72c446` である。

主な実行コマンド:

```bash
uv run pytest -q
uv run python scripts/run_ablations.py --conditions A0 --seeds 9 --output-root artifacts/issue9-phase3
uv run python scripts/run_ablations.py --jobs 6 --output-root artifacts/issue9-phase3
uv run python scripts/run_ablations.py --summarize-only --output-root artifacts/issue9-phase3
```

学習レポートは 36 件生成された。最初の A0 report から summary 生成までの artifact timestamp の区間は約 45 分 37 秒である。個別学習の wall clock は summary に記録されていない。

## 2. Ablation Matrix Results

> 初版の p 値・ラベルは、McNemar が A0 の未学習 baseline と比較していた不具合（`05911f1` で修正）により無効である。本表の p 値と discordant は修正後に `--summarize-only` を再実行して再集約した値であり、再学習は行っていない。
> ラベルはさらに、代表 seed 1 本ではなく**記録された全 seed が p<0.05**であることを要求する規則（PR #14 のレビュー指摘で変更）へ差し替えた。この規則変更で動いたのは A6 のみである（`significant_better` → `indeterminate`）。表中の p 値・discordant は再集約時のまま変わっていない。

`test error` は d1 test の `end_to_end_errors` で、3 seed の中央値と min-max を示す。

| condition | spec | test error median | min-max | McNemar p (seed 9/17/33) | discordant (seed 9/17/33) | label |
|---|---|---:|---:|---|---|---|
| A0 | `decomposed_v1` | 538 | 527-538 | 1 / 1 / 1 | 0 / 0 / 0 | 基準 |
| A1 | `decomposed_v1-lex_only` | 540 | 534-541 | .86 / .21 / .76 | 32 / 23 / 43 | `indeterminate` |
| A2 | `decomposed_v1-trans_only` | 566 | 561-566 | — | — | `indeterminate`（縮退） |
| A3 | `decomposed_v1-no_ipadic_cost` | 588 | 587-602 | 8.8e-34 / 1.8e-38 / 記録なし | 450 / 476 / 記録なし | `significant_worse` |
| A4 | `decomposed_v1-no_source` | 529 | 528-535 | .71 / 1 / .08 | 29 / 15 / 21 | `indeterminate` |
| A5 | `decomposed_v1-no_katakana` | 530 | 529-531 | .11 / .58 / .25 | 25 / 13 / 27 | `indeterminate` |
| A6 | `decomposed_v1-no_shape` | 520 | 519-522 | .035 / .43 / .081 | 73 / 57 / 74 | `indeterminate` |
| M0 | `decomposed_v1+matrix_v1` | 530 | 528-534 | .26 / .19 / .08 | 38 / 21 / 26 | `indeterminate` |
| M1 | `decomposed_v1+matrix_v1-pos_pairs` | 531 | 531-538 | .42 / .13 / .41 | 55 / 43 / 53 | `indeterminate` |
| F0 | `decomposed_v1+freq_v1` | 543 | 540-552 | .050 / .049 / .51 | 44 / 37 / 37 | `indeterminate` |
| F1 | `decomposed_v1+freq_v1-no_ipadic_cost` | 589 | 588-592 | 1.3e-7 / 8.8e-10 / 2.9e-7 | 86 / 109 / 95 | `significant_worse` |
| X0 | `decomposed_v1+matrix_v1+freq_v1` | 534 | 533-538 | .90 / .40 / .67 | 58 / 36 / 48 | `indeterminate` |

`indeterminate` は「差が無い」ではなく、d1 test 2,000 件では判別できないという意味である。McNemar は修正後の learned A0 と各条件の learned を同一 seed で比較した。不一致数が 25 未満の表は exact binomial、それ以上は continuity-corrected chi-square を使っている。A2 は zero-cost の探索 collapse、A3 seed 33 は downstream artifact が欠けたため、該当 p 値・discordant は記録に含まれていない。A3 のラベルは記録のある 2 seed に基づく。

ラベルは**最も弱い seed** で決めている。seed 間で変わるのはサンプルではなくモデルなので、3 seed 中 1 つでしか見えない差はその条件が再現する差ではない。3 seed の分割表を合算しないのも同じ理由で、同じ test 2,000 件を 3 回数えることになり対応データの独立性が壊れる。この規則により A6 は `indeterminate` である（seed 9 の p=.035 に対し seed 17 は .43、seed 33 は .081）。加えて本節は 11 条件 × 3 seed = 33 回の検定であり多重比較の補正をしていない。条件数 11 だけで補正しても閾値は約 .0045 なので、単独条件の .05 直下の p を強い証拠として読んではならない。判別できたのは A3 と F1 の悪化だけである。

`test error median` と `min-max` は `report.json` の `end_to_end_errors`、McNemar の分割表は `diagnose_d1.json` の converter top-1 正誤であり、測定が異なる。11 条件では両者が一致するが、A3 だけ乖離する。A3 の p 値は §4 の learned surface recall@1 `.602`（誤り 796 件）と A0 の 538 件との 258 件差を検定しており、中央値列の 588 対 538 の 50 件差ではない。方向は同じなのでラベルは変わらないが、A3 の 2 指標が 208 件乖離する理由は artifact から特定できておらず、未解決事項として §10 に記す。

既存 suite は採否判断に使っていない。handmade は 14 件で、1 件の変動が Top-1 を 7.1 pt 動かすためである。seed 9 の記録値は A0: handmade `13/14`、AJIMEE CER `0.2870918585668476`、Mozc `87/498`、X0: handmade `13/14`、AJIMEE CER `0.29239512203125634`、Mozc `89/498` である。A2 の suite evaluate は記録に含まれていない。

## 3. Learned Balance Table

以下は `lex.ipadic_cost_mapped=1` に正規化した seed 9 の記録値、および summary に記録された範囲である。IPADIC anchor 自体を除去する A2/A3/F1 は正規化値が記録できない。

| condition | matrix | katakana | reading logprob | reading count | overlay/source | prediction check |
|---|---:|---:|---:|---:|---:|---|
| A0 | 記録なし | 2.2223 | 記録なし | 記録なし | -8.0275 | baseline |
| A1 | 記録なし | 2.4520 | 記録なし | 記録なし | -2.4154 | 比較条件 |
| A4 | 記録なし | 2.0135 | 記録なし | 記録なし | 記録なし | source を除去した条件 |
| A5 | 記録なし | 記録なし | 記録なし | 記録なし | -8.1155 | katakana を除去した条件 |
| A6 | 記録なし | 0.9170 | 記録なし | 記録なし | -1.1803 | shape を除去した条件 |
| F0 | 記録なし | 1.3840 | -1.2704 | 2.6490 | -4.6656 | 頻度特徴を利用 |
| M0 | -0.1386 | 2.6161 | 記録なし | 記録なし | -4.1615 | matrix を利用 |
| M1 | -0.0465 | 1.5820 | 記録なし | 記録なし | -10.9812 | pair を除去 |
| X0 | 0.0475 | 1.0736 | -1.6234 | 2.9340 | -7.0489 | matrix+freq |

設計ノートの予測との答え合わせでは、matrix は M0/M1/X0 で利用されたが、正規化値は M0 `-0.1386`、X0 `0.0475` であり、「0.2 倍程度」という単一予測をそのまま確認できる結果ではない。katakana は多くの条件で正値だが、A6/X0 ではそれぞれ `0.9170`/`1.0736` で、少し不利という方向は一貫していない。F0/X0 は読み logprob と候補数の非ゼロ値を持ち、頻度特徴は利用された。overlay は A0/A1/A5/A6/F0/M0/M1/X0 で非ゼロだが、出所特徴の重要度を raw weight だけから断定していない。`lex.ipadic_cost_mapped` を除去した条件では anchor 正規化ができず、記録にない。

`lex.script.*` と `trans.*` は任意の完全経路について `Σ lex.script.* == |P| == Σ trans.*` を満たす（Mozc 200 読み × K-best 5 の 1,000 経路で照合・不一致 0）。特徴空間に厳密な線形従属が 1 本あるため、個々の重み値を特徴の重要度として読んではならない。採否判断は §2 のエラー率差で行った。

`lex.reading_len` は全条件で 0 のままである。`Σ len(token.reading) == len(reading)` により gold-pred 差分が常に 0 で、構造的に学習されないためである。

## 4. Oracle Recall@k

`beam_width=64`、`k=1,5,10,20,64`。下表は seed 9 の d1 test artifact に記録された baseline と learned の代表値である。A2 の oracle artifact は search collapse により記録に含まれていない。

| condition | lattice coverage | baseline path@1 / @64 | learned path@1 / @64 | baseline surface@1 / @64 | learned surface@1 / @64 |
|---|---:|---:|---:|---:|---:|
| A0 | 1.0 | .690 / 1.000 | .731 / 1.000 | .690 / 1.000 | .731 / 1.000 |
| A1 | 1.0 | .690 / 1.000 | .730 / 1.000 | .690 / 1.000 | .730 / 1.000 |
| A2 | 記録なし | 記録なし | 記録なし | 記録なし | 記録なし |
| A3 | 1.0 | .690 / 1.000 | .706 / .9995 | .690 / 1.000 | .602 / .9995 |
| A4 | 1.0 | .690 / 1.000 | .7325 / 1.000 | .690 / 1.000 | .7325 / 1.000 |
| A5 | 1.0 | .690 / 1.000 | .7355 / 1.000 | .690 / 1.000 | .7355 / 1.000 |
| A6 | 1.0 | .690 / 1.000 | .7405 / 1.000 | .690 / 1.000 | .7405 / 1.000 |
| M0 | 1.0 | .690 / 1.000 | .735 / .9995 | .690 / 1.000 | .735 / .9995 |
| M1 | 1.0 | .690 / 1.000 | .7345 / 1.000 | .690 / 1.000 | .7345 / 1.000 |
| F0 | 1.0 | .690 / 1.000 | .724 / 1.000 | .690 / 1.000 | .724 / 1.000 |
| F1 | 1.0 | .690 / 1.000 | .706 / 1.000 | .690 / 1.000 | .706 / 1.000 |
| X0 | 1.0 | .690 / 1.000 | .731 / 1.000 | .690 / 1.000 | .731 / 1.000 |

`unique_surface_count` の A0 seed 9 は baseline が mean `49.201` / p50 `52` / p90 `62` / max `64`、learned が mean `47.9415` / p50 `51` / p90 `61` / max `64` だった。X0 seed 9 は baseline が mean `49.201` / p50 `52` / p90 `62` / max `64`、learned が mean `46.631` / p50 `49` / p90 `60` / max `64` だった。Phase 4 の `rerank_k` はこの分布を使って決める。learned は 1-best だけでなく、多くの条件で recall@64 も維持したが、A3/M0 の recall@64 は `.9995` へ低下した。従って「1-best だけを上げた」とは結論できず、上位集合の測定が必要である。

AJIMEE は `convert_sentence` が文を分割して変換するため、単一読みの oracle recall 定義と一致せず対象外とした。

## 5. Stratified Results

A0 seed 9 の d1 diagnose artifact に記録された値を示す。baseline/learned は end-to-end error rate である。

| axis / layer | n | baseline | learned | low support |
|---|---:|---:|---:|---|
| reading length 1-3 | 637 | .4097 | .3501 | no |
| reading length 4-6 | 1187 | .2628 | .2359 | no |
| reading length 7-10 | 153 | .2745 | .2026 | no |
| reading length 11-16 | 23 | .2174 | .1739 | yes |
| proper noun false / true | 1751 / 249 | .3050 / .3454 | .2678 / .2771 | no |
| numeric false / true | 1413 / 587 | .2895 / .3595 | .2583 / .2947 | no |
| span_tokens 1 / 2 / 3 | 1730 / 143 / 127 | .2994 / .3986 / .3543 | .2694 / .2238 / .3150 | no |
| contrast pair false / true | 1872 / 128 | .2831 / .7031 | .2415 / .6719 | no |

`span_tokens` と contrast pair は MeCab 変換品質の代理指標であり、変換品質そのものの層別ではない。真の品質について記録されている根拠は d1 の 200 件抽出監査 `label_accuracy=0.995` のみである。件数 30 未満の reading length `11-16` は `low support` である。

## 6. Case-Level Before/After

handmade の全 14 件と Mozc の 498 件について、baseline/learned の全順位表は GPU 側の `diagnose_handmade.json` / `diagnose_mozc.json` に保存されている。A0 seed 9 の同一 baseline/learned 診断では handmade は `unchanged=13`、`still_wrong=1`、`fixed=0`、`regressed=0` だった。Top-1 が変わらないケースも省略していない。Mozc の全 498 件の表はこのレポート本文には展開せず、artifact を正本とする。なお A2/A3 の一部 diagnose artifact は search collapse/resource pressure のため記録に含まれていない。

`hashi-no` の記録は次のとおりである。

- baseline: `橋の`（gold rank 1、cost `1.2500000000000002`）
- learned: `橋野`（gold rank 2、Top-1 回帰）
- baseline-weight path difference: `+2.81`
- learned-weight path difference: `-0.08087461634929378`
- learned-weight の特徴寄与上位: `lex.source.overlay +3.006545000000375`、`lex.ipadic_cost_mapped +0.7603008836500251`、`lex.base_cost -1.690546499999977`、`lex.script.hiragana -1.3852774999997564`

従って Phase 2 の「Wikipedia 頻度・固有名詞バイアス」という仮説だけでは説明を完結できない。記録された寄与分解は、経路差を baseline 重みと learned 重みで分離し、overlay/source、IPADIC cost、base cost、script、transition の数値を示している。これらの寄与をさらに学習データの出所因果へ結び付ける測定は記録に含まれていない。

代表 fixed/regressed は d1 について summary の `representative` に ID 昇順で最大 20 件として記録されている。A2/A3 の欠損分を推測で補っていない。

## 7. Cost-Scale Recalibration

A0 seed 9 の d1 confidence 分布:

| | p10 | p25 | p50 | p75 | p90 | count |
|---|---:|---:|---:|---:|---:|---:|
| baseline | .0030 | .10875 | .5790 | 2.16125 | 4.07510 | 2000 |
| learned | .00112 | .03440 | .14711 | 2.45354 | 3.12161 | 2000 |

baseline threshold `2.0` の fire rate は `.7395`、learned threshold は `2.025162545128425`、scale ratio は `.2540704446070626`、p25 ratio は `.3163106328966384`、p75 ratio は `1.1352410719817418`、初期 `viterbi_weight` は `3.9359162831653594` である。handmade self-check は scale ratio `1.0`、threshold `2.0`、初期倍率 `1.0` だった。Mozc の同一重み self-check も scale ratio `1.0`、threshold `2.0` である。AJIMEE の分割単位は 250 件だった。

`needs_rerank` は `converter.py:174` で計算されるが、実際の発火は `:176` の context/reranker 有無だけで決まり、参照は `evaluation.py:250` の集計だけである。従って threshold の再較正は統計量の測り直しであり、精度改善ではない。この配線は Phase 3 では変更していない。`viterbi_weight` の実較正は実リランカが必要なため Phase 4 に送った。Phase 3 は候補間 confidence 分布から初期倍率を出しただけである。

## 8. Leakage Audit

miniPC 側の audit artifact では manifest consistency が `all_passed=true`、Mozc source-document claim は `verified=true` だった。Mozc の記録フィールドは `id`、`label`、`reading`、`expected`、`command`、`expected_rank` で、原文/context 列は無かった。

d1 JSONL が無い環境での audit だったため、d1 と評価データの document n-gram overlap、reading overlap、train vocabulary overlap、article ID による split field 再計算は `unverifiable` と記録されている。したがってこれらを「リークなし」とは結論しない。reading overlap と vocabulary overlap は、測定できた場合も document leakage とは別の分布重複として扱う設計である。GPU 側でこの完全監査を実行した結果は記録に含まれていない。

## 9. Negative and Null Results

- A3 と F1 は `significant_worse`。A1 は indeterminate、F0 も境界的な p 値（seed 9 が .050、seed 17 が .049、seed 33 が .51）で indeterminate である。F0 のラベルは .05 のすぐ両側に 2 seed が並ぶため、しきい値の位置に依存する。
- A2 は transition-only の zero-cost tie collapse が起き、AJIMEE evaluate は候補空/探索長時間化で完走しなかった。これは想定された縮退下限であり、推測値を補っていない。
- **改善側で有意と言える条件は 1 つも無い。** A6 は 3 seed 中 seed 9 のみ p=.035（seed 17 は .43、seed 33 は .081）で、全 seed 一致を要求する規則では indeterminate である。33 回の検定に対して補正もしていないため、この p を改善の証拠として扱わない。A4/A5/M0/M1/X0 も中央値の方向は改善だが indeterminate であり、2,000 件では判別できない。既存 suite の小標本値は採否根拠にしていない。
- 従って Phase 3 で判別できたのは **A3（IPADIC word cost の除去）と F1（頻度による word cost の代替）が悪化させること** だけである。「どの特徴を足せば良くなるか」については、この標本サイズでは何も言えていない。
- F2（Wikipedia corpus frequency）は実施していない。外部 corpus が無く、リーク監査を先に必要とするためである。
- 実リランカを使った `viterbi_weight` の実較正も実施していない。実モデルは GPU 側に無く、Phase 4 scope である。
- Task 8 後の A2/A3 rerun では A2 3 seed と A3 seed 33 の downstream artifact が欠けた。記録された failed 一覧は summary の `failed` を正本とし、欠損を成功値として扱っていない。

## 10. Limitations and Phase 4 Handoff

1. 改善側の条件がすべて indeterminate であるため、PLAN §11.1 のルールに従い **A0 を既定とする**。A6/M0/M1/X0 はいずれも中央値の方向は改善だが判別できていないので、優劣を根拠に選ばない。A2/A3 の collapse と d1 complete audit 不在も考慮する。
2. 改善側を判別するには標本を増やすしかない。A6 の seed 9 の効果量（2,000 件で 18〜19 件・discordant 73）を再現性のあるものとして確認したいなら、d1 test の拡張か、複数 seed をまたいで安定する評価設計が Phase 4 の前提になる。
3. **未解決**: A3 だけ `end_to_end_errors`（588）と diagnose の converter top-1 誤り（796）が 208 件乖離している（§2 末尾）。他 11 条件では一致するため A3 固有の現象であり、探索 collapse 下で `evaluate_test` と `diagnose_cases` の測定条件が食い違っている可能性がある。artifact で特定すること。
4. surface recall 曲線は d1 artifact の `beam_width=64` 結果を使う。A0 は learned `.731` at 1 / `1.0` at 64、X0 は `.731` at 1 / `1.0` at 64 だった。頭打ちの正確な unique-surface 分布は記録に含まれていないため、rerank_k をこのレポートだけから決めない。
5. A0 seed 9 の cost-scale ratio は `.2540704446070626`、従って初期 `viterbi_weight` は `3.9359162831653594`。実リランカで再較正する。
6. `cascade_threshold` は現状 gate ではない。Phase 4 で実際のカスケード gate として配線するか判断する。
7. reading length 1-3、numeric true、contrast pair true、proper noun true などの層別値は d1 test A0 artifact の弱点候補である。span_tokens/contrast は品質そのものではなく代理指標である。
8. leakage audit は d1 JSONL 不在のため完全検証されていない。GPU 側の完全監査を実施してから評価データへの非リークを主張する。
9. package-data の結論は Phase 3 の scope 外であり、学習済み weights は外部 path 必須と明示する。package-data は `legacy_v1.json` と `decomposed_v1.json` のみに限定する方針を Phase 4 に引き継ぐ。
10. feature spec v2 では `lex.reading_len` の削除、script/transition 共線性の解消、高 cardinality 語彙特徴、外部疎 weights を検討する。
11. matrix cost は位置非依存なので Phase 3 では `LatticePath` を再設計していない。位置依存特徴を追加する場合は、`LinearScorer._edge_cache` の token key と path の位置保持を同時に再設計する。
