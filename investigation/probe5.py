import sys, math, heapq, time, tracemalloc
from itertools import count
sys.path.insert(0, "/home/sarashin/work/beme-p2/src")
from beme.dictionary import Dictionary
from beme.lattice import build_lattice, backward_viterbi, pos_transition_cost, LatticePath
from beme.benchmark import load_ajimee_cases, katakana_to_hiragana

d = Dictionary.from_tsv("/home/sarashin/work/beme-p2/data/dictionary.tsv")
cases = load_ajimee_cases("/home/sarashin/work/beme-p2/data/ajimee_bench_jwtd_v2_v1.json")
r = katakana_to_hiragana(cases[2].input)
lat = build_lattice(r, d, allow_literals=True)
remaining = backward_viterbi(lat)
term = len(lat.nodes) - 1

tracemalloc.start()
seq = count(); q = []
heapq.heappush(q, (remaining[0], 0.0, "", tuple(), next(seq), 0, LatticePath(0.0)))
res = []; pops = 0
PUSH_CAP = 300_000
while q and len(res) < 64:
    _,_,_,_,_,pos,path = heapq.heappop(q); pops += 1
    if pos == term: res.append(path); continue
    prev = path.token.pos if path.token is not None else None
    for e in lat.nodes[pos].edges:
        if math.isinf(remaining[e.end]): continue
        c = path.cost + e.token.cost + pos_transition_cost(prev, e.token.pos)
        child = LatticePath(c, e.token, path)
        tk = tuple((t.reading,t.surface,t.pos,t.cost) for t in child.tokens)
        heapq.heappush(q, (c+remaining[e.end], c, child.surface, tk, next(seq), e.end, child))
    if len(q) >= PUSH_CAP: break
cur, peak = tracemalloc.get_traced_memory()
print(f"heap entries: {len(q)}  pops: {pops}")
print(f"tracemalloc current: {cur/1e6:.1f} MB  peak: {peak/1e6:.1f} MB")
print(f"=> bytes per heap entry: {cur/max(len(q),1):.0f}")
print(f"observed max heap in full run was ~1.0M entries -> projected {cur/max(len(q),1)*1_000_000/1e9:.2f} GB")
tracemalloc.stop()

# how many LatticePath objects have a materialized _tokens_cache?
cached = sum(1 for it in q if it[-1]._tokens_cache is not None)
print(f"\npaths on heap with _tokens_cache materialized: {cached}/{len(q)} ({100*cached/len(q):.0f}%)")
avg_len = sum(len(it[-1]._tokens_cache) for it in q if it[-1]._tokens_cache) / max(cached,1)
print(f"avg cached token tuple length: {avg_len:.1f}")

# lattice composition (M4 literal edges)
lit = sum(1 for n in lat.nodes for e in n.edges if e.token.pos == "literal")
unk = sum(1 for n in lat.nodes for e in n.edges if e.token.pos == "unknown")
tot = sum(len(n.edges) for n in lat.nodes)
print(f"\nlattice: len={len(r)} total_edges={tot} literal={lit} unknown={unk} dict={tot-lit-unk}")
print(f"avg dict edges per hiragana position: {(tot-lit-unk)/max(len(r)-lit,1):.1f}")
