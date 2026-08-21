import sys, math, heapq, time
from itertools import count
sys.path.insert(0, "/home/sarashin/work/beme-p2/src")
from beme.dictionary import Dictionary
from beme.lattice import build_lattice, backward_viterbi, pos_transition_cost, LatticePath
from beme.benchmark import load_ajimee_cases, katakana_to_hiragana

d = Dictionary.from_tsv("/home/sarashin/work/beme-p2/data/dictionary.tsv")
cases = load_ajimee_cases("/home/sarashin/work/beme-p2/data/ajimee_bench_jwtd_v2_v1.json")

def exact_remaining(lat):
    """R[pos][prev_pos] = true min remaining cost given last token POS."""
    n = len(lat.nodes); term = n - 1
    R = [dict() for _ in range(n)]
    poss = set()
    for node in lat.nodes:
        for e in node.edges: poss.add(e.token.pos)
    poss.add(None)
    for pos in range(n - 1, -1, -1):
        if pos == term:
            for p in poss: R[pos][p] = 0.0
            continue
        for p in poss:
            best = math.inf
            for e in lat.nodes[pos].edges:
                nxt = R[e.end].get(e.token.pos, math.inf)
                if math.isinf(nxt): continue
                c = pos_transition_cost(p, e.token.pos) + e.token.cost + nxt
                if c < best: best = c
            R[pos][p] = best
    return R

def cost_stats(lat):
    cs = {}
    for node in lat.nodes:
        for e in node.edges:
            cs[e.token.cost] = cs.get(e.token.cost, 0) + 1
    return cs

for i in [0, 2, 3, 1]:
    r = katakana_to_hiragana(cases[i].input)
    lat = build_lattice(r, d, allow_literals=True)
    h = backward_viterbi(lat)
    R = exact_remaining(lat)
    # gap between heuristic h[p] and the exact remaining for the *best* prev_pos
    gaps = []
    for p in range(len(lat.nodes) - 1):
        exact_best = min(v for v in R[p].values() if not math.isinf(v)) if R[p] else math.inf
        exact_worst = max((v for v in R[p].values() if not math.isinf(v)), default=math.inf)
        if math.isinf(h[p]) or math.isinf(exact_best): continue
        gaps.append((p, round(h[p],4), round(exact_best,4), round(exact_worst,4)))
    print(f"\ncase {i} len={len(r)} edges={sum(len(n.edges) for n in lat.nodes)}")
    print("  h[0]=%.4f  exact_from_BOS=%.4f  gap=%.4f" % (h[0], R[0][None], R[0][None]-h[0]))
    mg = max(g[2]-g[1] for g in gaps)
    print("  max gap over positions: %.4f" % mg)
    print("  sample (pos, h, exact_best, exact_worst):", gaps[:6])
    cs = cost_stats(lat)
    print("  distinct edge costs:", len(cs), "top:", sorted(cs.items(), key=lambda kv:-kv[1])[:6])
