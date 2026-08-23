## Phase 4 完了報告（投稿草案）

> このファイルは issue #9 へのコメント草案です。まだ投稿していません。

Phase 4（文脈モデルへの接続）を完了しました。結論は PLAN §5 の **「効いた」分岐**です。d1 test の contrast pair 128件で、局所モデル + Cross-Encoder の併用が局所モデル単独より有意に改善し、seed 9/17/33 の全てで同じ方向でした。

### チェックリスト

- [x] **局所モデルの K-best を #2 / #5 のリランカへ渡す**
  - `score`（CE relevanceを1回計算）と `combine`（結合比・k・gateを推論ゼロで集計）の2段に分離しました。
  - #5 のD1小型LMは既存評価でNo-Goのため、Phase 4は `hotchpotch/japanese-reranker-base-v2` を実測しました。
- [x] **oracle recall と Top-1 改善を分けて評価する**
  - `surface_recall@10` は全体 `.988000`、contrast `.960938` でリランク前後不変、2,000件のhead集合違反は0でした。
  - Top-1の実績は全体 `+.100500`、contrast `+.203125` で、候補上限と実績を別指標として記録しました。
- [x] **局所モデル単独 / リランカ単独 / 併用を比較する**
  - devだけで `viterbi_weight=8`, `rerank_k=10` を選び、testは選択後に評価しました。

### d1 test 3条件比較

A0 seed 9、全体2,000件 / contrast 128件です。

| 条件 | 定義 | overall Top-1 | contrast Top-1 | contrast McNemar `b/c/p` |
|---|---|---:|---:|---:|
| L | 局所cost順 | .731000 | .328125 | 0/0/1.0 |
| R | 候補生成はLと同一、順位だけrelevance（`vw=0`, `k=64`） | .521000 | .421875 | 18/30/.112351 |
| LR | 局所cost + relevance（`vw=8`, `k=10`） | **.831500** | **.531250** | 7/33/`7.72e-5` |

Rのoverallは有意に悪化しました。Rのcontrastは改善方向ですが、128件では有意改善と判別できません。LRはoverall/contrastともに改善方向・有意でした。

seed安定性:

| seed | overall before→after | contrast before→after |
|---:|---:|---:|
| 9 | .7310→.8315 | .328125→.531250 |
| 17 | .7365→.8320 | .328125→.531250 |
| 33 | .7310→.8310 | .328125→.531250 |

`rerank_stats.label` はoverall/contrastとも `significant_better` です。単一seedやseed間でpoolした分割表ではラベルを付けていません。mock対照は `b=c=0`, `p=1.0` でした。

### 設計上の決定

1. **Top-1定義の統一**: 学習評価とConverterを `(cost, surface)` 最小へ統一しました。Phase 3でA3だけ208件ずれていた実バグを解消し、同点のない11条件は不変でした。
2. **2段分離**: CE relevanceは `(context, surface)` の関数なので一度だけscoreし、`viterbi_weight × rerank_k × gate` はoffline combineしました。
3. **「リランカ単独」の定義**: IMEでは候補生成を省けないため、候補生成はLと同じで、順位付けだけをrelevanceにしました。
4. **primaryをcontrast pairにした理由**: 全体2,000件には文脈で順位を変える必要の薄い例が多く、文脈識別力を直接測る128件をprimaryにしました。dev contrastは44件しかないため、設定選択自体はdev全体1,000件を主指標、contrastを副指標にしました。

### レイテンシとcascade gate

Intel N100・4 core miniPC（host `WIN-I7DHB6T7L79`）のd1 test先頭50件、モデルロード除外の実測です。

| 設定 | 平均/変換 | contrast 128 Top-1 |
|---|---:|---:|
| 全候補 | 3.321963 s | .523438 |
| `rerank_k=10` | .739371 s | .531250 |

`rerank_k=10` は77.74%削減し、contrast Top-1も低下しませんでした。この差自体には統計検定をしていないので、k=10のほうが精度も高いとは判定していません。

cascade候補はdevで評価しましたが、`recall≥.95` を満たす最小の登録済み条件 `confidence<.25` がfire rate `.567`、test確認でも`.576`でした。停止基準`.5`を超えたため、**「ゲートでは削減できない」と結論して配線していません**。k=10 + gateの期待値`.428 s/変換`は反実仮想で、実配線値ではありません。

### リーク監査

d1を持つGPU環境でPhase 3の未検証項目を再実行しました。

- `manifest all_passed=true`
- split再計算 `verified=true`, mismatches `[]`
- failures `[]`, unverifiable `[]`
- AJIMEE/handmadeのdocument n-gram match 0
- Mozcは原文/context fieldなしを再確認
- Mozc reading共有 30/465、vocabulary共有 9/481。ただしこれはdocument leakageではなく分布重複として記録

「あらゆるリークがない」ではなく、監査定義と保持fieldの範囲でdocument matchなし・split整合、という結論です。

### レビューで確認した実バグ・残リスク

- Phase 3で見つかったA3の208件乖離は、Top-1の同点規則が学習評価とConverterで違う実バグでした。Phase 4 Task 1で共有入口へ統一しました。
- Phase 3のMcNemar比較対象取り違えと代表seed 1本ラベルを再発させないため、`paired_table` はID集合不一致/重複を例外化し、`label` は複数seed全ての有意・同方向を要求します。
- Task 3後の別モデルcross-reviewでは重大不具合なし。将来 `span_tokens` の固定bucketを呼び出し側が仮定すると `KeyError` になり得る軽微な残リスクが指摘されましたが、現契約違反ではないため変更していません。

### Phase 5 / 次の作業への注意点

- `vw=8` はdev gridの上端。追加探索する場合もtestを再選択に使わないこと。
- cascade thresholdを追試するなら、今回の事後的な`.12997`をそのまま採用せず、新しいdev/testまたはnested splitで先に固定すること。
- LRが効いた分岐なので、X0（matrix + frequency）とCEの組合せ再測定を候補にすること。
- loss 54件には同音異義語だけでなく人名・表記揺れがあるため、#2の頻度バイアス、#3のcase分類へ引き継ぐこと。
- AJIMEEで文脈効果を測るには、現在の文分割評価とは別に文脈供給設計が必要です。
- 別issue候補: 学習済みweightのpackage-data/外部path方針、contrast・人名・表記揺れを含む評価セット拡張。

### テスト

Phase 3完了時 `585 passed` → Phase 4完了時:

```text
613 passed in 73.56s
```

詳細レポート: `docs/missions/issue9-phase4/phase4-report.md`

artifact正本: `artifacts/issue9-phase4/`（各主要artifactのSHA-256はレポート§1）
