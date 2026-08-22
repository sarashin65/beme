# P0 実装後の oracle recall 再測定（AJIMEE 200 件）

測定日: 2026-08-21  
対象: BeME `ba66606ad1c88997bd18e15285bdd7959c9d3824`（P0: カタカナ表層コスト +1.0）

## 1. 結論

P0 後の AJIMEE 200 件における **5-best oracle recall は 28/200 = 14.0%** だった。P0 前の **26/200 = 13.0%** から、差は **+2 件 / +1.0 percentage point（相対 +7.7%）** に留まる。

一方、Top-1 は 12/200（6.0%）から 14/200（7.0%）へ改善した。したがって、5-best 内の正解を完全なリランカで Top-1 に昇格できる件数は、P0 前後とも **14 件**で変わらない。

- P0 前: 5-best 正解 26 件 − Top-1 正解 12 件 = 昇格可能 14 件（+7.0 points）
- P0 後: 5-best 正解 28 件 − Top-1 正解 14 件 = 昇格可能 14 件（+7.0 points）

**[INFERENCE] 固定 5-best のまま Phase 2 リランカだけへ投資する場合、天井問題は実質的に解消していない。** P0 は CER を大きく改善したが、正解全文を候補集合へ新たに入れる効果は小さかった。

ただし、同じ P0 後モデルで候補幅を広げる補助測定では、oracle recall は `k=10` で 19.5%、`k=20` で 23.5%、`k=64` で 31.5% まで上がった。**[INFERENCE] P2 の全面実装へ直行するより、まず既存候補生成の候補幅を広げて実リランカ・品質・レイテンシを測るオフライン gate には投資価値がある。**

## 2. 定義と測定条件

### 2.1 oracle recall

`docs/deepdive-2026-08-21.md` §2.5 と同じく、各ケースについて `convert_sentence(..., k=5)` が返す文候補の `surface` のいずれかが、AJIMEE の許容正解 `references` のいずれかと**全文完全一致**すれば hit とした。

$$
\mathrm{oracle\ recall@5}
= \frac{\#\{i \mid \exists c \in C_i^{(5)},\ \exists r \in R_i:\ c=r\}}{200}
$$

これは deepdive §1.2 の「正解経路がラティス上に存在するか」を調べるオラクル DP（P0 前 175/200 = 87.5%）とは別の指標である。本レポートでいう oracle recall は、Phase 2 が実際に受け取れる**文 5-best 内への正解全文の包含率**を指す。

### 2.2 固定条件

- 対象 HEAD: `ba66606ad1c88997bd18e15285bdd7959c9d3824`
- AJIMEE: `data/ajimee_bench_jwtd_v2_v1.json` の全 200 件
- 辞書: `data/dictionary.tsv`（P0 コスト反映済み）
- `beam_width=64`
- 文候補 `k=5`
- `cascade_threshold=2.0`
- `preceding_context=case.context`
- 既存の `~/.cache/beme-diag/diag.py` を再実行

比較元の 13.0% は deepdive §2.5 の 26/200。再実行前に残っていた `diag.json` も 26/200、Top-1 12/200、失敗分類 12 correct / 25 coverage / 60 model-homophone / 101 model-segmentation / 2 search で、deepdive の前回測定と一致した。これを `~/.cache/beme-diag/diag-before-p0.json` に退避して差分を取った。

## 3. 再現手順

作業ディレクトリは `~/work/beme`。

```sh
# 対象コミット
cd ~/work/beme
git rev-parse HEAD
# ba66606ad1c88997bd18e15285bdd7959c9d3824

# P0 後の基準指標
.venv/bin/python scripts/evaluate.py --suite ajimee --json

# deepdive §8 と同じ診断。各行の top5 が今回の主測定データ
.venv/bin/python ~/.cache/beme-diag/diag.py
# -> ~/.cache/beme-diag/diag.json

# diag.json から oracle recall@5 と正解順位を集計
.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

rows = json.loads(Path.home().joinpath('.cache/beme-diag/diag.json').read_text())
ranks = []
for row in rows:
    rank = next(
        (i for i, surface in enumerate(row['top5'], 1)
         if surface in row['refs']),
        None,
    )
    ranks.append(rank)
hits = sum(rank is not None for rank in ranks)
print({'cases': len(rows), 'hits': hits, 'recall': hits / len(rows)})
print({'rank_histogram': dict(sorted(Counter(r for r in ranks if r).items()))})
PY
# {'cases': 200, 'hits': 28, 'recall': 0.14}
# {'rank_histogram': {1: 14, 2: 10, 3: 1, 4: 1, 5: 2}}

# 補助測定: 同じ beam_width=64 で k を 64 まで広げた感度分析
.venv/bin/python ~/.cache/beme-diag/oracle_recall_after_p0.py
# -> ~/.cache/beme-diag/oracle-recall-after-p0.json
```

`oracle_recall_after_p0.py` は各ケースを `k=64` で一度生成し、正解の最小順位から各 cutoff を集計する。`k=64` 出力の先頭 5 件と `diag.py` の `top5` は **200/200 件で一致**した。

## 4. 主測定結果

### 4.1 P0 前後比較

| 指標 | P0 前 | P0 後 | 差 |
|---|---:|---:|---:|
| mean CER | 0.341433 | 0.290080 | −0.051353（相対 −15.0%） |
| mean edit distance | 6.56 | 5.53 | −1.03 |
| Top-1 完全一致 | 12/200 (6.0%) | 14/200 (7.0%) | +2件 / +1.0 point |
| **oracle recall@5** | **26/200 (13.0%)** | **28/200 (14.0%)** | **+2件 / +1.0 point** |
| 5-best 内・Top-1 外の正解 | 14/200 (7.0%) | 14/200 (7.0%) | 変化なし |
| 5-best miss | 174/200 (87.0%) | 172/200 (86.0%) | −2件 |

CER と edit distance は `scripts/evaluate.py --suite ajimee --json` の実測値。P0 は CER を相対 15.0% 改善した一方、oracle recall@5 の改善は 1.0 point だった。

**[INFERENCE] P0 の主効果は、正解全文を多数新規生成することではなく、カタカナ混じりの不自然な上位経路を下げて Top-1 の文字列を正解へ近づけることにある。** CER 改善量と全文一致候補の増加量の乖離がこれを示す。

### 4.2 正解順位分布

| 正解の最小順位 | P0 前 | P0 後 | 差 |
|---:|---:|---:|---:|
| 1 | 12 | 14 | +2 |
| 2 | 9 | 10 | +1 |
| 3 | 2 | 1 | −1 |
| 4 | 2 | 1 | −1 |
| 5 | 1 | 2 | +1 |
| 合計 | 26 | 28 | +2 |

P0 前の hit 26 件のうち 25 件は P0 後も残った。集合差は **新規 3 件、脱落 1 件**。両方で hit した 25 件は、順位改善 3 件、同順位 22 件、悪化 0 件だった。

### 4.3 P0 により 5-best 包含が変わったケース

| ID | 正解 | P0 前 → 後 | 観測した候補変化 |
|---|---|---:|---|
| `ajimee-jwtd-v2-1029` | `度を超える`（許容正解の一つ） | miss → 5位 | `ドを…` 候補が下がり、正解が 5 位へ入った |
| `ajimee-jwtd-v2-1559` | `意見が分かれて` | miss → 1位 | `意見が若レテ` / `意見がワカレテ` より正解が上位化 |
| `ajimee-jwtd-v2-881` | `漢字を和訓に当てている由来` | miss → 5位 | `…アテて…` 候補が下がり、正解が 5 位へ入った |
| `ajimee-jwtd-v2-1233` | `ニューギャラリーの開館にスウェーデンを代表して` | 4位 → miss | 正しいカタカナ語を含む候補が落ち、`Νｅｗギャラリー…` 系が 5-best を占めた |

最後の脱落ケースの正解は、P0 後の `k=64` では **18 位**に残っていた。P0 は不自然な `ド`、`レテ`、`アテ` を抑える一方、正しい外来語表記にも同じ固定ペナルティを課すという §5.5 記載の trade-off が候補集合でも確認できた。

Top-1 完全一致の新規 2 件は `ajimee-jwtd-v2-1559` と `ajimee-jwtd-v2-1581`（`のでスグに分かる` → `ので直ぐに分かる`）。Top-1 完全一致からの脱落は 0 件だった。

### 4.4 オラクル DP の補助観測

`diag.py` の P0 後分類は次の通り。

| 分類 | P0 前 | P0 後 |
|---|---:|---:|
| correct | 12 | 14 |
| coverage | 25 | 25 |
| model-homophone | 60 | 65 |
| model-segmentation | 101 | 94 |
| search | 2 | 2 |

coverage 25 件と search 2 件は不変。P0 は辞書エントリを削除せずコストだけを変えるため、ラティス上の正解経路の有無が変わらないことと整合する。model の細分類変化は、最良仮説とオラクル経路の境界比較が P0 後の仮説に応じて変わった結果である。

## 5. 候補幅の感度分析（P2 の判断材料）

主測定は従来定義どおり `k=5`。P2 の投資判断用に、同じ P0 後辞書・同じ `beam_width=64` で `convert_sentence` の `k` だけを最大 64 へ広げ、正解の最小順位を集計した。

| cutoff | 正解を含む件数 | oracle recall |
|---:|---:|---:|
| 1 | 14 | 7.0% |
| 5 | 28 | 14.0% |
| 10 | 39 | 19.5% |
| 20 | 47 | 23.5% |
| 50 | 59 | 29.5% |
| 64 | 63 | 31.5% |

`k=64` で返った候補数は最小 31、最大 64、平均 59.955。`beam_width=64` の保持経路内だけの測定であり、ラティス全体の理論上限ではない。

入力を `_input_spans` の変換可能なひらがな連続区間数で分けると次の通り。

| 変換可能 span 数 | ケース数 | hit@5 | hit@64 |
|---:|---:|---:|---:|
| 1 | 104 | 25 | 49 |
| 2 | 53 | 3 | 13 |
| 3 | 26 | 0 | 1 |
| 4 | 11 | 0 | 0 |
| 5 | 2 | 0 | 0 |
| 6 | 4 | 0 | 0 |
| 合計 | 200 | 28 | 63 |

5-best から 64-best へ広げて回収した 35 件の内訳は、single-span 24 件、multi-span 11 件だった。

- **観測**: 正解全文は現在の 5-best より深い順位に相当数残る。候補幅を 64 へ広げるだけで包含率は 14.0% から 31.5% へ上がった。
- **観測**: multi-span は 96 件中 hit@5 が 3 件、hit@64 でも 14 件。single-span より低い。
- **[INFERENCE]** multi-span の低さは P2 の問題設定と一致するが、multi-span 文の長さ・難しさが交絡する。この集計だけでは、スパン合成時の枝刈りが原因だとは断定できない。
- **[INFERENCE]** 追加 35 件中 24 件は single-span なので、その部分は「真の文 K-best 作り直し」を待たず、候補幅を広げて文リランカへ渡す PoC で検証できる。

## 6. P2 への判断

### 6.1 固定 5-best の Phase 2 追加投資

**現時点では優先しない。** P0 後も oracle ceiling は 14.0%に過ぎず、完全なリランカでも Top-1 完全一致を現在の 7.0%から最大 14.0%へ上げるだけである。しかも昇格可能件数 14 件は P0 前と同じだった。

### 6.2 文レベル K-best の全面的な作り直し

**直ちに production 実装する根拠はまだ不足。ただし、候補集合を広げるオフライン PoC は実施価値あり。**

理由:

1. P0 による recall@5 の純増は 2 件だけで、P0 が天井問題を自然に解消する期待は否定された。
2. 一方で hit@64 は 63 件あり、現行コストの beam 保持範囲にも 5 位より下の正解が 35 件ある。候補生成・候補受け渡しを改善する余地は実測で存在する。
3. その 35 件のうち 24 件は single-span。全面的な文 K-best 合成の再設計より先に、既存生成器から `k=10/20/64` を文リランカへ渡す実験で、実際に何件を Top-1 へ回収できるかを測れる。
4. 正しいカタカナ語を含む `ajimee-jwtd-v2-1233` は P0 後も 18 位に残る。候補幅拡大は P0 の副作用をリランカで回収する経路にもなる。

**推奨 gate:**

- AJIMEE を tuning 用と評価用に分離し、`k=5/10/20/64` で実リランカ後の CER・完全一致・レイテンシを比較する。
- single-span / multi-span を分けて、候補幅拡大だけで得られる分と文合成再設計でしか得られない分を切り分ける。
- 実リランカが oracle 増分を十分回収し、かつレイテンシを満たした場合に限り、スパン境界コストを含む真の文 K-best へ進む。

**[INFERENCE] 現在の証拠が支持する優先順位は「固定 5-best リランカ追加」ではなく、「候補幅を広げたオフライン gate」→「効果が出た場合のみ P2 全面実装」である。**

## 7. 制約と解釈上の注意

- AJIMEE 200 件は P0 の +1.0 を選ぶ際にも使われた。同じ集合上の改善であり、汎化性能の証拠ではない。
- 13.0% の比較元は deepdive の前回測定と、その時点から残っていた `diag.json`。旧コミットを checkout して再実行はしていない。
- `k=64` 感度分析は `beam_width=64` が保持した経路に制約される。31.5% を候補生成の理論上限とはみなさない。
- oracle recall は「候補にあれば必ず選べる」という上限値。実リランカの回収率、CER、レイテンシは今回測定していない。
- P2 の新しい文コスト、スパン境界接続、枝刈り方式は未実装。`k` 感度分析は P2 実装効果そのものではなく、投資判断用の候補集合感度である。
- オラクル DP のパッチ長上限など deepdive §9 の制約は失敗分類には残るが、`top5` の全文完全一致を数える主指標には影響しない。

## 8. 生成物

リポジトリ内:

- `docs/oracle-recall-after-p0-2026-08-21.md`（本レポート）

リポジトリ外の測定データ:

- `~/.cache/beme-diag/diag-before-p0.json` — 前回 13.0% の候補データ退避
- `~/.cache/beme-diag/diag.json` — P0 後の `diag.py` 出力
- `~/.cache/beme-diag/oracle_recall_after_p0.py` — recall@k・span 数別集計
- `~/.cache/beme-diag/oracle-recall-after-p0.json` — 補助測定サマリ

production コードは変更していない。git commit / push は行っていない。
