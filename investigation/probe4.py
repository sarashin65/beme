import sys, math
sys.path.insert(0, "/home/sarashin/work/beme-p2/src")
from beme.lattice import Lattice, LatticeNode, LatticeEdge, Token, backward_viterbi

def N(p, *e): return LatticeNode(p, tuple(e))
def E(s, e, sur, po, c): return LatticeEdge(s, e, Token(sur, sur, po, c))

# node1 has two "noun" edges of DIFFERENT length:
#   A: cost 10, spans 1->4 (terminal)  -> after_edge(A) = 0
#   b: cost  1, spans 1->2, then must pay c(9)+d(9)  -> after_edge(b) = 18.16
# continuation[1]["noun"] = min(0, 18.16) = 0  <-- collapses both into one slot
lat = Lattice("wxyz", (
    N(0, E(0,1,"w","noun",0.0)),
    N(1, E(1,4,"A","noun",10.0), E(1,2,"b","noun",1.0)),
    N(2, E(2,3,"c","noun",9.0)),
    N(3, E(3,4,"d","noun",9.0)),
    N(4),
))
h = backward_viterbi(lat)
print("backward_viterbi ->", [round(x,3) for x in h])

# ground truth by brute force
import itertools
from beme.lattice import pos_transition_cost
def brute(pos, prev):
    if pos == 4: return 0.0
    best = math.inf
    for e in lat.nodes[pos].edges:
        best = min(best, pos_transition_cost(prev, e.token.pos) + e.token.cost + brute(e.end, e.token.pos))
    return best
print("true remaining from pos0 (BOS) =", round(brute(0, None), 3))
print("h[0] =", round(h[0], 3), " -> UNDERESTIMATE by", round(brute(0,None) - h[0], 3))
print()
print("Why: h[0] evaluates edge w by  min over node1's edges of")
print("   next.cost + trans + continuation[1][next.pos]")
print("   A -> 10.0 + 0.08 + 0.0 = 10.08")
print("   b ->  1.0 + 0.08 + 0.0 =  1.08   <-- b借用了 A 的 continuation(0.0)")
print("   picks 1.08, but b's real continuation is 18.16 (true best is A at 10.08)")
