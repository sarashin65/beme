import sys, time, math, heapq
from itertools import count
sys.path.insert(0, "/home/sarashin/work/beme-p2/src")
from beme.dictionary import Dictionary
from beme.lattice import build_lattice, backward_viterbi, pos_transition_cost, LatticePath
from beme.benchmark import load_ajimee_cases, katakana_to_hiragana

d = Dictionary.from_tsv("/home/sarashin/work/beme-p2/data/dictionary.tsv")
cases = load_ajimee_cases("/home/sarashin/work/beme-p2/data/ajimee_bench_jwtd_v2_v1.json")
print("cases:", len(cases))
lens = sorted(len(c.input) for c in cases)
print("input len: min", lens[0], "med", lens[len(lens)//2], "max", lens[-1])

def instrumented_kbest(lattice, K, pop_limit=2_000_000, time_limit=20.0):
    remaining = backward_viterbi(lattice)
    if not remaining or math.isinf(remaining[0]):
        return None
    seq = count(); queue = []
    root = LatticePath(0.0)
    heapq.heappush(queue, (remaining[0], 0.0, "", tuple(), next(seq), 0, root))
    results = []; terminal = len(lattice.nodes) - 1
    pops = pushes = maxheap = 0
    t0 = time.perf_counter()
    while queue and len(results) < K:
        if pops > pop_limit or time.perf_counter() - t0 > time_limit:
            return dict(done=False, pops=pops, pushes=pushes, maxheap=maxheap,
                        results=len(results), secs=round(time.perf_counter()-t0, 2))
        _, _, _, _, _, position, path = heapq.heappop(queue)
        pops += 1
        if position == terminal:
            results.append(path); continue
        prev = path.token.pos if path.token is not None else None
        for edge in lattice.nodes[position].edges:
            if math.isinf(remaining[edge.end]):
                continue
            cost = path.cost + edge.token.cost + pos_transition_cost(prev, edge.token.pos)
            child = LatticePath(cost, edge.token, path)
            tk = tuple((t.reading, t.surface, t.pos, t.cost) for t in child.tokens)
            heapq.heappush(queue, (cost + remaining[edge.end], cost, child.surface, tk,
                                   next(seq), edge.end, child))
            pushes += 1
        if len(queue) > maxheap:
            maxheap = len(queue)
    return dict(done=True, pops=pops, pushes=pushes, maxheap=maxheap,
                results=len(results), secs=round(time.perf_counter()-t0, 2))

for i in [0, 1, 2, 3, 4]:
    c = cases[i]
    r = katakana_to_hiragana(c.input)
    t = time.perf_counter()
    lat = build_lattice(r, d, allow_literals=True)
    bt = time.perf_counter() - t
    ne = sum(len(n.edges) for n in lat.nodes)
    print(f"\ncase {i} len={len(r)} nodes={len(lat.nodes)} edges={ne} build={bt:.3f}s")
    print("  input:", r[:60])
    for K in (5, 64):
        print(f"  K={K}: {instrumented_kbest(lat, K)}")
