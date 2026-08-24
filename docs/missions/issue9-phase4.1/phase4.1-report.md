# Issue #17 Phase 4.1 Report

## 1. 結論

事前登録ルール 1 の seed 増比較により、**A8（`decomposed_v1-no_script-surface_len`）を新ベースとして採用する**。A7 は seed 1/5/9/13/17/21/25/29/33/37 の全 10 seed で A8 を上回り、A8 < A7 は 10/10 seed、差分は `Δ=-1〜-12`、両側符号検定は `p=.001953`（<.05）だった。これは事前登録どおりの AMEND である。

A8 と A6 は既存の 3 seed 比較でともに中央値 520 であり、記述統計上は同性能だった。A8 は A6（`decomposed_v1-no_shape`）と同じ script / surface length 除去を行いながら `lex.proper_noun` を保持する。A8 と A6 の直接 McNemar は判別不能であり、単一 seed の p 値から差のラベルは付けない。

従って、A8 採用の根拠は A7/A8 の 10 seed 増比較による AMEND であり、「proper_noun 保持の効果」や surface length の因果的寄与まで確定したものではない。

## 2. Scope and Reproducibility

実装・全テスト・4 指標回帰は miniPC `WIN-I7DHB6T7L79`（Intel N100）で実行した。9 run の学習・評価・集計は GPU マシン `sarashin`（Arch Linux、AMD Ryzen 9 5900X 12-Core / 24 CPU、NVIDIA GeForce RTX 3060 Ti 8192 MiB、driver 610.57.04）の `/home/sarashin/beme-p41` で実行した。実行 commit は `2314dc763265e03700ce349cae50197bb686ac7d`、親は `aabd0571685e140821219d665b119ddd043971cb`、branch は `feat/issue9-phase4.1` である。GPU 側は uv 0.12.5 / CPython 3.14.7 を使った。

A0 は Phase 3 の `/home/sarashin/beme-p3/artifacts/issue9-phase3/A0` を参照した。A6/A7/A8 は同日・同 commit・同マシンで再学習した。d1 は `/home/sarashin/beme-d1/repo/data/d1/{train,dev,test}.jsonl` を clone 内へ symlink した。

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

主な実行コマンド:

```bash
gh issue view 17
uv run python scripts/build_ablation_weights.py
uv run pytest -q tests/test_feature_registry.py tests/test_run_ablations.py
uv run pytest -q
git diff --check

git bundle create /tmp/beme-4p1.bundle feat/issue9-phase4.1
scp /tmp/beme-4p1.bundle ws:/tmp/beme-4p1.bundle
ssh ws 'git clone --branch feat/issue9-phase4.1 /tmp/beme-4p1.bundle ~/beme-p41'
ssh ws '~/.local/bin/uv sync --directory ~/beme-p41'

# GPU 側で nohup + setsid、PPID=1 を確認して実行
~/.local/bin/uv run python scripts/run_ablations.py \
  --conditions decomposed_v1-no_script,decomposed_v1-no_script-surface_len,decomposed_v1-no_shape \
  --seeds 9,17,33 \
  --output-root artifacts/issue9-phase4.1
~/.local/bin/uv run python scripts/run_ablations.py \
  --summarize-only --output-root artifacts/issue9-phase4.1

uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json --json
```

学習プロセス PID 919885 は detachment 後に `PPID=1` であることを確認した。9 run は全件完走し、summary の `failed=[]`、各 McNemar の `compared=2000`、`unpaired=0` だった。

## 3. 実装とテスト

`src/beme/features.py` の既存 `_ABLATION_REMOVALS` パターンへ次を追加した。

| condition | feature spec | 除去 | 保持 |
|---|---|---|---|
| A7 | `decomposed_v1-no_script` | `lex.script.{kanji,hiragana,katakana,other}` | `lex.surface_len`, `lex.proper_noun` |
| A8 | `decomposed_v1-no_script-surface_len` | script 4 特徴 + `lex.surface_len` | `lex.proper_noun` |

`build_ablation_weights.py` は registry を列挙する既存設計なので、重複する条件表は追加せず、registry 追加から 2 個の sparse 初期重み JSON を生成した。A7/A8 の `lex.surface_len` / `lex.proper_noun` は初期値 0 なので JSON の `weights` map には現れないが、feature spec の宣言には残る。テストは extractor 出力で script 4 特徴の不在、A7 の surface length / proper noun の存在、A8 の surface length 不在 / proper noun 存在を直接確認する。strict weights load の既存 parameterized test も A7/A8 の生成 JSON を検証する。

`scripts/run_ablations.py` には A7/A8 の label を追加し、mission に記載された feature spec ID 形式の `--conditions` を既存の A6/A7/A8 label へ正規化した。既存 label 入力は維持される。
今回の 10 seed 集計時点では `summarize_only()` が `SEEDS=(9,17,33)` をハードコードしており、`--summarize-only --seeds` を無視する制限があった。そのため 10 seed の error は GPU artifact の各 `report.json` の `selected_model.test.end_to_end_errors` を直接抽出して符号検定した。この制限は本 AMEND で `--seeds` を `summarize_only()` に渡すよう修正し、指定 seed だけを集計する回帰テストを追加した。

検証結果:

- targeted harness: `20 passed` (`tests/test_run_ablations.py`)
- final full suite: `640 passed in 77.13s`
- `git diff --check`: exit 0
- 探索コア `src/beme/lattice.py` / `converter.py` は変更していない

初期重み SHA-256:

| initial weights | SHA-256 |
|---|---|
| `decomposed_v1-no_script.json` | `ab55b9e71408cddb797070611533529e5ee691ba875f5a055db4bdedfaf1c0ef` |
| `decomposed_v1-no_script-surface_len.json` | `741f5d505ece5756e7ac01ee9c904a9d96636e5e4f94c6eeaaaf12ef235c2946` |
| `decomposed_v1-no_shape.json` | `dcbe671bbf4f3977e4e23193dab912a054c649fd7d728c31dad556a7fc3d2297` |

## 4. A6 / A7 / A8 × A0 と seed 増比較

`test error` は d1 test 2,000 件の `end_to_end_errors`。`b` は A0 correct / condition wrong、`c` は A0 wrong / condition correct である。

### 4.1 既存 3 seed の A0 比較

| condition | error seed 9/17/33 | median | min-max | b/c seed 9/17/33 | McNemar p seed 9/17/33 | summary label |
|---|---|---:|---:|---|---|---|
| A0 | 538 / 527 / 538 | 538 | 527-538 | 0/0, 0/0, 0/0 | 1 / 1 / 1 | 基準 (`indeterminate`) |
| A6 | 519 / 520 / 520 | 520 | 519-520 | 27/46, 25/32, 27/45 | .035140 / .426777 / .045127 | `indeterminate` |
| A7 | 530 / 532 / 529 | 530 | 529-532 | 16/24, 11/6, 10/19 | .268382 / .332306 / .137395 | `indeterminate` |
| A8 | 521 / 520 / 520 | 520 | 520-521 | 29/46, 23/30, 26/44 | .064672 / .409847 / .042165 | `indeterminate` |

この 3 seed の A0 比較は、全条件が `indeterminate` である。これは差が無いという意味ではなく、事前登録済みの「全 seed が `p<.05` かつ中央値の方向が一致」というラベル規則では判別できないという意味である。A6 seed 9/33、A8 seed 33 の単一 seed p 値から有意改善ラベルを付けていない。3 seed の表は同じ test 2,000 件を評価するため pool していない。

### 4.2 A7/A8 seed 増比較（10 seed）

| condition | seed 1 | seed 5 | seed 9 | seed 13 | seed 17 | seed 21 | seed 25 | seed 29 | seed 33 | seed 37 | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A7 | 531 | 524 | 530 | 527 | 532 | 529 | 528 | 531 | 529 | 532 | 529.5 |
| A8 | 520 | 523 | 521 | 518 | 520 | 519 | 523 | 520 | 520 | 525 | 520.0 |
| A8 - A7 | -11 | -1 | -9 | -9 | -12 | -10 | -5 | -11 | -9 | -7 | — |

| 比較 | A8 < A7 | 差分 | 両側符号検定 | 判定 |
|---|---:|---|---:|---|
| A8 vs A7 | 10/10 seed | `Δ=-1〜-12` | `p=.001953` | 事前登録どおり AMEND |

A8 は 10/10 seed で A7 を下回った。A6 は 3 seed の既存値 `519/520/520`（中央値 520）のまま、A8 の中央値 520.0 と記述統計上は同性能である。A8 は A6 と同じ script / surface length 除去を行いながら `lex.proper_noun` を保持する。10 seed 比較は seed ごとの条件間 error を比較した両側符号検定であり、test 2,000 件を pool していない。単一 seed の p 値でラベル付けしていない。

## 5. 3 点の直接比較（既存 3 seed）

直接比較の `b/c` は表の左条件 correct / 右条件 wrong と、その逆である。

| comparison | b/c seed 9/17/33 | McNemar p seed 9/17/33 | 判定 |
|---|---|---|---|
| A6 → A7 | 29/18, 30/18, 27/18 | .144661 / .112351 / .233038 | A7 は記述上悪い方向だが判別不能 |
| A7 → A8 | 18/27, 16/28, 20/29 | .233038 / .097254 / .253098 | A8 は記述上良い方向だが判別不能 |
| A8 → A6 | 0/2, 3/3, 4/4 | .500000 / 1 / 1 | 差は判別不能 |

記述統計では、script だけを除く A7（中央値 530）より surface length も除く A8（中央値 520）の方向が良く、A8 と A6（中央値 520）は同性能だった。A6 は `proper_noun` を除くが、A8 は `proper_noun` を保持する。しかし直接 McNemar は全比較で判別不能であり、proper_noun 保持や surface length 除去の効果を単一 seed p 値でラベル付けしない。A7/A8 の 10 seed 符号検定（p=.001953）が今回の AMEND の根拠である。

## 6. 事前登録ルール 6 条

| # | 事前登録ルール | 実測判定 |
|---:|---|---|
| 1 | A7 ≥ A0: 有意悪化なし、中央値 ≤ A0 なら A7 | **AMEND**。旧 3 seed では A7 median 530 ≤ 538 でルール 1 の採用条件を満たしていたが、seed 増比較で A8 が A7 を 10/10 seed 下回り、両側符号検定 `p=.001953` となった。事前登録どおり A8 に AMEND し、A8 を新ベースにする。 |
| 2 | A7 < A6 が有意なら A6 | **不成立**。方向は A7 が悪いが、A6→A7 p=.145/.112/.233 で有意差を判別できない。A6 override なし。 |
| 3 | A7 ≥ A6 なら A7 優先 | **判別不能**。中央値は A7 が悪い方向で、統計的優劣も付かない。この規則を採用根拠には使わない。 |
| 4 | 人名3件 + proper-noun 244件を確認 | **実施済み**。§7。3 人名は全 condition / 全 seed で正解。244件層は seed 9 の before 66 errors に対し A6/A7/A8=63/64/63。 |
| 5 | 4 指標回帰を bit-exact 維持 | **成立**。§8。 |
| 6 | A8/A6 採用時、CE `vw=8/k=10` を3 seed確認 | **発火・ハンドオフ要**。A8 用 CE score/combine は本 Phase 4.1 の local-model runには含めておらず、`LR ≥ A0` は未判定。A0 cost 前提の既存 CE artifactを流用して合格扱いにはしない。 |

ルール 6 は新ベース A8 の Phase 4 CE 選択点再確認として残る。A8 seed 9/17/33 の各 weights で CE score を再生成し、固定 `viterbi_weight=8`, `rerank_k=10` の combine を同 seed の A0 と比較する必要がある。

## 7. ケース確認

### 7.1 人名 3 件

表の各セルは seed 9/17/33 の learned Top-1。全セルが gold と一致した。

| id | reading | gold | A0 | A6 | A7 | A8 |
|---|---|---|---|---|---|---|
| `5536f0fb6a3d07d3766e` | しゅんいち | 俊一 | 俊一 / 俊一 / 俊一 | 俊一 / 俊一 / 俊一 | 俊一 / 俊一 / 俊一 | 俊一 / 俊一 / 俊一 |
| `9042ddd6ad1ad6911122` | えいさく | 栄作 | 栄作 / 栄作 / 栄作 | 栄作 / 栄作 / 栄作 | 栄作 / 栄作 / 栄作 | 栄作 / 栄作 / 栄作 |
| `f77b9a8a3a0e72ef6ec6` | きよひと | 清仁 | 清仁 / 清仁 / 清仁 | 清仁 / 清仁 / 清仁 | 清仁 / 清仁 / 清仁 | 清仁 / 清仁 / 清仁 |

Phase 4 の CE 併用で観測された `俊一→駿一` / `栄作→英作` / `清仁→清人` は、今回の局所モデル A6/A7/A8 では再現しなかった。従って、この3件から A7/A8 の proper noun 保持効果を条件間差として実証したとは言えない。一方、A6 でも3件が壊れなかったため「A6 は必ずこの3件を壊す」とも言えない。

### 7.2 proper-noun 244 件

Phase 4 と同じ244件層を再現するため、A0 seed 9 learned weights を baseline scorer とし、その constrained gold path に `proper_noun` token があるケースを抽出した。A0 seed 9 の before error は66件（`.270492`）。同じ244件・seed 9 condition の after を比較した。

| condition | n | before errors/rate | after errors/rate | net |
|---|---:|---:|---:|---:|
| A6 | 244 | 66 / .270492 | 63 / .258197 | -3 errors |
| A7 | 244 | 66 / .270492 | 64 / .262295 | -2 errors |
| A8 | 244 | 66 / .270492 | 63 / .258197 | -3 errors |

これは case audit の単一 seed 記述値であり、有意ラベルには使わない。244件層では3条件とも net regression は観測されず、A7/A8 の proper noun 保持が A6 より大きい net gain を生む結果でもなかった。gross gain/loss の層内内訳は診断 artifact の現行 schema に記録されないため、net before/after 以上は判別不能である。

## 8. 4 指標回帰

default A0 (`data/weights/decomposed_v1.json`) を変更していないことを `evaluate.py --suite all --reranker mock` で実測した。

| metric | expected | actual | result |
|---|---:|---:|---|
| AJIMEE mean CER | 0.290080 | 0.29008046985634645 | bit-exact JSON value / 6桁一致 |
| AJIMEE exact match | 0.070000 | 0.07 | 一致 |
| Mozc passed/evaluated | 88/498 | 88/498 | 一致 |
| handmade Top-1 | 1.000000 | 1.0 | 一致 |

4 指標回帰は不変である。学習済み A6/A7/A8 自体の suite 値は base implementation regression ではなく、採用判断にも使っていない。

## 9. Artifact SHA-256

学習済み weights / diagnostics / summary は gitignore 対象で、git には追加しない。GPU 正本は `/home/sarashin/beme-p41/artifacts/issue9-phase4.1/`、解析用コピーは local `artifacts/issue9-phase4.1/` にある。

| artifact | SHA-256 |
|---|---|
| `summary.json` | `beb10498beefc2f6857542ff80e47ff8086d6804fa9b30bc1fb5a18943226eb6` |
| `phase41-analysis.json` | `95259f824516967e1dea9d772fa921a507e0343a6c937a4e91395c3c915b53f0` |
| `regression/evaluate_all_mock.json` | `1c2d142a33135556582ddcf961cd527eca81935e5f91c259d2ba2ce87fa8c20e` |
| `A6/seed9/diagnose_proper244.json` | `19258371d5c200b8bf03b4c18a4e464d9570b0d8e6325d895d486e0df51a9db2` |
| `A7/seed9/diagnose_proper244.json` | `e5add36363d06e530de3ed938682602a87d51ff560cfdec921a8faad989180a3` |
| `A8/seed9/diagnose_proper244.json` | `09a82bb194ef990a81c36f4232128088a61944e68c3afe77773535faefcc54cd` |
| A6 seed 9/17/33 weights | `0ba6b4ec4e72d10ffa2bbb9f861c2577ff5ea98d38e2f9151724eae25e53b1a6` / `e8fe673f9c3c741e2216a0b8395ea7652961c99990dd6ec6eaf082b016e9998c` / `dedef59eee7bf27494f1d524ccd30ab448eba6c3b08392270eca166c676cbf7d` |
| A7 seed 9/17/33 weights | `72091cdc5ccd7917bd37af41808b05d0acc0e1a535b69248ecff9e38c366e41f` / `d80be82a93ca22df047b83062851bc94b02dad7e73f043c965812f27cbbe1880` / `37af39a4749b968a2849224f0aaece067b526189a064f9d5d4eb98110074bb20` |
| A8 seed 9/17/33 weights | `2b03c7f2c1e665fc05dabfb1c2d5f557f5cbfd941111fd727a271e15ac9ab15c` / `692f4c628907db5a374bf09db0e63ad2dfddd26a965e914df6730e52c33f6e1e` / `8325fc6fb3f8fcdfc0d009231960aece3ec45951a3bcfd4ddda981f0f302422e` |

## 10. 受け入れ条件

| acceptance criterion | result |
|---|---|
| A7/A8 feature spec と初期重み生成、宣言特徴除去テスト | pass |
| d1 test 3 seed × A6/A7/A8 完走、summary に A0 比較を記録 | pass (`failed=[]`) |
| 事前登録ルールによる新ベース決定 | A8。seed 増比較で 10/10 seed、両側符号検定 `p=.001953` により事前登録どおり AMEND |
| 4 指標回帰 bit-exact | pass |
| 判別不能・単一 seed p 非採用・artifact hash・実行機を記録 | pass |

残件は事前登録ルール 6 の CE 3 seed handoff である。A8 は A6 と同じ中央値（既存 3 seed の記述統計）であり、`proper_noun` を保持する。10 seed の A7/A8 比較により A8 を新ベースへ AMEND した。
