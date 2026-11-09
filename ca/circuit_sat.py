"""Direct SAT synthesis of a small 2-input-gate circuit CA rule under the exact-rollout constraint.

Circuit family C(n, G): pool = [tap inputs 0..n-1, gate 0, ..., gate G-1]. Gate j reads pool
signals a < b (both < n + j) and applies an arbitrary 2-input function tt (bit q = output for
(a, b) = (q >> 1, q & 1), the costs.rule_circuit / circuit-v1 convention). Output channel c
selects any pool signal. Symmetry handling v1: a < b, and every gate is used by a later gate or an
output (no dangling gates; buffer padding keeps SAT monotone in G). v2 adds lexicographic ordering of
adjacent independent gates (see `lex`).

State bits: constants where known (boundary, observed visible frames every S generations, hidden
init at g = 0), variables otherwise. One circuit copy per distinct tuple of input literals.
"""


class CircuitSAT:
    def __init__(self, frames, taps, channels, G, S=1, boundary=0, hidden_init=0, lex=False, entries=False):
        """entries=False (v1/v2 'direct'): one circuit copy per cell-generation input tuple.
        entries=True (v3): the rollout is encoded over LUT-entry variables by ca.sat.Encoder (as in
        exp03), and the circuit is evaluated once per LUT entry present in that CNF, on constant
        inputs, with its output tied to the entry variable. Same solution set: every consulted entry
        equals the circuit's value; unconsulted entries are free in both."""
        self.F, self.taps, self.C, self.G, self.S = frames, taps, channels, G, S
        self.boundary, self.hidden_init, self.lex, self.entries = boundary, hidden_init, lex, entries
        self.n = len(taps); self.nv = 0; self.cl = []; self.X = {}; self.cache = {}
        n = self.n
        if entries:
            from ca.sat import Encoder
            self.enc = Encoder(taps, channels, S, boundary, hidden_init, sink=self.cl.append).extend(frames)
            self.nv = self.enc.nv; self.X = self.enc.X
        self.TRUE = self._new(); self.cl.append([self.TRUE])
        self.tt = [[self._new() for _ in range(4)] for _ in range(G)]
        self.sa = [[self._new() for _ in range(n + j)] for j in range(G)]
        self.sb = [[self._new() for _ in range(n + j)] for j in range(G)]
        self.so = [[self._new() for _ in range(n + G)] for _ in range(channels)]
        for j in range(G):
            self._exactly_one(self.sa[j]); self._exactly_one(self.sb[j])
            for p in range(n + j):          # a < b
                for q in range(p + 1):
                    self.cl.append([-self.sa[j][p], -self.sb[j][q]])
        for c in range(channels):
            self._exactly_one(self.so[c])
        for j in range(G):                  # every gate is used
            self.cl.append([self.sa[k][n + j] for k in range(j + 1, G)] + [self.sb[k][n + j] for k in range(j + 1, G)]
                           + [self.so[c][n + j] for c in range(channels)])
        if lex:
            self._lex()
        self._entries() if entries else self._rollout()

    def _entries(self):
        for (c, e), v in self.enc.L.items():
            pool = self._eval([self.TRUE if (e >> t) & 1 else -self.TRUE for t in range(self.n)])
            for s, p in zip(self.so[c], pool):
                self.add([-s, -v, p]); self.add([-s, v, -p])

    def _new(self):
        self.nv += 1; return self.nv

    def add(self, c):
        """Clause with constant folding: drop if it contains TRUE, remove -TRUE literals."""
        if self.TRUE in c:
            return
        c = [l for l in c if l != -self.TRUE]
        assert c, "empty clause: constant contradiction"
        self.cl.append(c)

    def _exactly_one(self, xs):
        self.cl.append(list(xs))
        for i in range(len(xs)):
            for k in range(i + 1, len(xs)):
                self.cl.append([-xs[i], -xs[k]])

    def _lex(self):
        """v2: for adjacent gates j, j+1 where j+1 does not read gate j, require
        (a_j, b_j) <= (a_{j+1}, b_{j+1}) lexicographically (they could be swapped otherwise)."""
        n = self.n
        for j in range(self.G - 1):
            k = j + 1  # clauses below apply only if gate k does not read gate j
            used_k = [self.sa[k][n + j], self.sb[k][n + j]]
            for p in range(n + j):
                for p2 in range(p):   # a_k = p2 < a_j = p  -> forbidden when independent
                    self.cl.append([-self.sa[j][p], -self.sa[k][p2]] + used_k)
                for q in range(n + j):
                    for q2 in range(q):  # a equal and b_k < b_j -> forbidden when independent
                        self.cl.append([-self.sa[j][p], -self.sa[k][p], -self.sb[j][q], -self.sb[k][q2]] + used_k)

    def lit(self, g, c, y, x):
        """Literal for state bit (generation g, channel c, y, x): TRUE/-TRUE for constants."""
        T, H, W = self.F.shape
        if not (0 <= y < H and 0 <= x < W):
            return self.TRUE if self.boundary else -self.TRUE
        if c == 0 and g % self.S == 0:
            return self.TRUE if self.F[g // self.S, y, x] else -self.TRUE
        if c > 0 and g == 0:
            return self.TRUE if self.hidden_init else -self.TRUE
        if (g, c, y, x) not in self.X:
            self.X[g, c, y, x] = self._new()
        return self.X[g, c, y, x]

    def _mux(self, sel, pool):
        """Fresh variable w with sel[p] -> (w <-> pool[p])."""
        w = self._new()
        for s, p in zip(sel, pool):
            self.add([-s, -w, p]); self.add([-s, w, -p])
        return w

    def _eval(self, ins):
        """Pool literals of one circuit copy on input literals `ins` (shared per distinct tuple)."""
        key = tuple(ins)
        if key in self.cache:
            return self.cache[key]
        pool = list(ins)
        for j in range(self.G):
            a = self._mux(self.sa[j], pool); b = self._mux(self.sb[j], pool); v = self._new()
            for q in range(4):
                ga = [-a if q >> 1 else a, -b if q & 1 else b]  # negation of (a == q>>1 and b == q&1)
                t = self.tt[j][q]
                self.add(ga + [-t, v]); self.add(ga + [t, -v])
            pool.append(v)
        self.cache[key] = pool
        return pool

    def _rollout(self):
        T, H, W = self.F.shape
        for g in range((T - 1) * self.S):
            for y in range(H):
                for x in range(W):
                    pool = self._eval([self.lit(g, c, y + dy, x + dx) for c, dy, dx in self.taps])
                    for c in range(self.C):
                        o = self.lit(g + 1, c, y, x)
                        for s, p in zip(self.so[c], pool):
                            self.add([-s, -o, p]); self.add([-s, o, -p])

    def decode(self, model):
        pos = set(v for v in model if v > 0)
        one = lambda xs: [i for i, v in enumerate(xs) if v in pos][0]
        gates = [(sum(1 << q for q in range(4) if self.tt[j][q] in pos), one(self.sa[j]), one(self.sb[j])) for j in range(self.G)]
        return gates, [one(s) for s in self.so]

    def hidden_trajectory(self, model):
        pos = set(v for v in model if v > 0)
        return {f"{g},{c},{y},{x}": int(v in pos) for (g, c, y, x), v in self.X.items()}
