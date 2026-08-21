import sys, math, heapq, time
from itertools import count
sys.path.insert(0, "/home/sarashin/work/beme-p2/src")
import beme.lattice as L
from beme.dictionary import Dictionary
from beme.converter import Converter
from beme.benchmark import load_ajimee_cases, evaluate_ajimee, load_mozc_cases, evaluate_mozc

def exact_remaining(lat):
    n = len(lat.nodes); term = n - 1
    poss = {None}
    for node in lat.nodes:
        for e in node.edges: poss.add(e.token.pos)
    R = [dict() for _ in range(n)]
    for p in poss: R[term][p] = 0.0
    for pos in range(n - 2, -1, -1):
        Rp = R[pos]; edges = lat.nodes[pos].edges
        for p in poss:
            best = math.inf
            for e in edges:
                nxt = R[e.end].get(e.token.pos, math.inf)
                if nxt == math.inf: continue
                c = L.pos_transition_cost(p, e.token.pos) + e.token.cost + nxt
                if c < best: best = c
            Rp[p] = best
    return R

def fixed_kbest(lattice, beam_width=64):
    R = exact_remaining(lattice)
    term = len(lattice.nodes) - 1
    h0 = R[0].get(None, math.inf)
    if math.isinf(h0): return tuple()
    seq = count(); q = [(h0, 0.0, next(seq), 0, L.LatticePath(0.0))]
    res = []
    while q and len(res) < beam_width:
        _, _, _, pos, path = heapq.heappop(q)
        if pos == term: res.append(path); continue
        prev = path.token.pos if path.token is not None else None
        for e in lattice.nodes[pos].edges:
            hh = R[e.end].get(e.token.pos, math.inf)
            if math.isinf(hh): continue
            c = path.cost + e.token.cost + L.pos_transition_cost(prev, e.token.pos)
            heapq.heappush(q, (c + hh, c, next(seq), e.end, L.LatticePath(c, e.token, path)))
    return tuple(res)

import beme.converter as C
C.k_best_viterbi = fixed_kbest

conv = Converter(Dictionary.from_tsv("/home/sarashin/work/beme-p2/data/dictionary.tsv"))
cases = load_ajimee_cases("/home/sarashin/work/beme-p2/data/ajimee_bench_jwtd_v2_v1.json")
t = time.perf_counter()
r = evaluate_ajimee(conv, cases, k=5)
dt = time.perf_counter() - t
print(f"AJIMEE 200 cases: {dt:.2f}s  mean_cer={r['mean_cer']:.4f} exact={r['exact_match_accuracy']:.4f}")

mz = load_mozc_cases("/home/sarashin/work/beme-p2/data/mozc_oss.tsv")
t = time.perf_counter()
rm = evaluate_mozc(conv, mz, k=5)
dt = time.perf_counter() - t
print(f"MOZC {rm['case_count']} cases ({rm['evaluated_count']} evaluated): {dt:.2f}s pass_rate={rm['pass_rate']:.4f}")
