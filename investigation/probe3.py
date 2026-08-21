import sys, math, heapq, time
from itertools import count
sys.path.insert(0, "/home/sarashin/work/beme-p2/src")
from beme.dictionary import Dictionary
from beme.lattice import build_lattice, backward_viterbi, pos_transition_cost, LatticePath
from beme.benchmark import load_ajimee_cases, katakana_to_hiragana

d = Dictionary.from_tsv("/home/sarashin/work/beme-p2/data/dictionary.tsv")
cases = load_ajimee_cases("/home/sarashin/work/beme-p2/data/ajimee_bench_jwtd_v2_v1.json")

def exact_remaining(lat):
    n = len(lat.nodes); term = n - 1
    poss = {None}
    for node in lat.nodes:
        for e in node.edges: poss.add(e.token.pos)
    R = [dict() for _ in range(n)]
    for p in poss: R[term][p] = 0.0
    for pos in range(n - 2, -1, -1):
        for p in poss:
            best = math.inf
            for e in lat.nodes[pos].edges:
                nxt = R[e.end].get(e.token.pos, math.inf)
                if math.isinf(nxt): continue
                c = pos_transition_cost(p, e.token.pos) + e.token.cost + nxt
                if c < best: best = c
            R[pos][p] = best
    return R

def astar(lat, K, mode, time_limit=25.0):
    """mode: 'loose+heavy' (current), 'loose+light', 'exact+light', 'exact+heavy'"""
    term = len(lat.nodes) - 1
    if mode.startswith("exact"):
        R = exact_remaining(lat)
        h = lambda pos, prev: R[pos].get(prev, math.inf)
    else:
        bw = backward_viterbi(lat)
        h = lambda pos, prev: bw[pos]
    heavy = mode.endswith("heavy")
    seq = count(); q = []
    root = LatticePath(0.0)
    h0 = h(0, None)
    if math.isinf(h0): return None
    heapq.heappush(q, (h0, 0.0, "", tuple(), next(seq), 0, root) if heavy else (h0, 0.0, next(seq), 0, root))
    res = []; pops = pushes = maxheap = 0
    t0 = time.perf_counter()
    while q and len(res) < K:
        if time.perf_counter() - t0 > time_limit:
            return dict(done=False, pops=pops, pushes=pushes, maxheap=maxheap, got=len(res),
                        secs=round(time.perf_counter()-t0,2))
        item = heapq.heappop(q); pops += 1
        pos, path = item[-2], item[-1]
        if pos == term:
            res.append(path); continue
        prev = path.token.pos if path.token is not None else None
        for e in lat.nodes[pos].edges:
            hh = h(e.end, e.token.pos)
            if math.isinf(hh): continue
            c = path.cost + e.token.cost + pos_transition_cost(prev, e.token.pos)
            child = LatticePath(c, e.token, path)
            if heavy:
                tk = tuple((t.reading, t.surface, t.pos, t.cost) for t in child.tokens)
                heapq.heappush(q, (c + hh, c, child.surface, tk, next(seq), e.end, child))
            else:
                heapq.heappush(q, (c + hh, c, next(seq), e.end, child))
            pushes += 1
        maxheap = max(maxheap, len(q))
    return dict(done=True, pops=pops, pushes=pushes, maxheap=maxheap, got=len(res),
                secs=round(time.perf_counter()-t0,3), best=round(res[0].cost,4) if res else None,
                top=res[0].surface if res else None)

for i in [0, 2, 3, 1]:
    r = katakana_to_hiragana(cases[i].input)
    lat = build_lattice(r, d, allow_literals=True)
    print(f"\n=== case {i} len={len(r)} edges={sum(len(n.edges) for n in lat.nodes)}")
    for mode in ("loose+heavy", "loose+light", "exact+heavy", "exact+light"):
        print(f"  {mode:12s} K=64 ->", astar(lat, 64, mode))
