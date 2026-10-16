"""Gap-based exact and approximate algorithms for circular-convex bicliques.

This strengthens the initial anchored minimum-union baseline. Nonempty
forbidden arcs fit into the gaps between consecutive chosen B vertices.
Their contribution functions have the single-crossing property. A segment
tree for their upper envelope accelerates the dynamic programs.

Only nonnegative integer vertex weights with product-of-sums objective are
supported. Representation is supplied as cyclic Arc(start, length) rows.
"""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction

from solver import Arc, Result, validate


@dataclass(frozen=True)
class State:
    last: int
    score: int  # A-weight credited to already completed open gaps
    mass: int   # actual B-weight of chosen vertices
    path: object


def gap_matrix(arcs, q, aw):
    """C[i][j] = total A-weight of forbidden arcs wholly inside (i,j).

    Points are duplicated at positions 0,...,2q-1. Only queries with
    0 < j-i <= q are used as circular gaps. Two copies of each nonempty
    proper forbidden arc suffice. Empty forbidden arcs are credited once
    as a universal-row offset; full forbidden arcs can never fit a gap.
    Construction: O(p+q^2) arithmetic operations and O(q^2) memory.
    """
    size = 2 * q
    hist = [[0] * size for _ in range(size)]
    universal = 0
    for arc, profit in zip(arcs, aw):
        if arc.length == q:
            universal += profit
        elif arc.length:
            first = (arc.start + arc.length) % q
            length = q - arc.length
            for lo in (first, first + q):
                hi = lo + length - 1
                if hi < size:
                    hist[lo][hi] += profit
    matrix = [[0] * (size + 1) for _ in range(size)]
    ending = [0] * size
    for i in range(size - 1, -1, -1):
        if i + 1 < size:
            add = hist[i + 1]
            for r, value in enumerate(add):
                ending[r] += value
        cumulative = 0
        for j in range(1, size + 1):
            cumulative += ending[j - 1]
            matrix[i][j] = cumulative
    return universal, matrix


class Envelope:
    """Li Chao-style maximum tree for the single-crossing gap functions.

    A function represented by State(i,s,...) has value s+C[i][x]. For
    i<j, the difference f_i(x)-f_j(x) is nondecreasing in x, because it
    counts forbidden intervals starting in (i,j] and ending before x,
    plus a constant. Any two functions therefore switch order at most
    once. Ties prefer larger mass, then smaller endpoint, then larger
    stored score. States identical on these keys represent identical
    functions and masses, so either witness is interchangeable.
    These priorities are constant over the query domain. Linear functions are
    not required. Inserts and point queries use O(log(q+1)) evaluations.
    """

    def __init__(self, matrix, lo, hi):
        self.matrix = matrix
        self.lo, self.hi = lo, hi
        self.nodes = {}

    def better(self, first, second, x):
        if first is None:
            return False
        if second is None:
            return True
        first_key = (first.score + self.matrix[first.last][x], first.mass, -first.last, first.score)
        second_key = (second.score + self.matrix[second.last][x], second.mass, -second.last, second.score)
        return first_key > second_key

    def insert(self, state):
        lo, hi, node = self.lo, self.hi, 1
        while True:
            current = self.nodes.get(node)
            if current is None:
                self.nodes[node] = state
                return
            mid = (lo + hi) // 2
            if self.better(state, current, mid):
                self.nodes[node], state = state, current
            if lo == hi:
                return
            stored = self.nodes[node]
            if self.better(state, stored, lo):
                node, hi = node * 2, mid
            elif self.better(state, stored, hi):
                node, lo = node * 2 + 1, mid + 1
            else:
                return

    def query(self, x):
        lo, hi, node = self.lo, self.hi, 1
        best = None
        while True:
            current = self.nodes.get(node)
            if self.better(current, best, x):
                best = current
            if lo == hi:
                return best
            mid = (lo + hi) // 2
            if x <= mid:
                node, hi = node * 2, mid
            else:
                node, lo = node * 2 + 1, mid + 1


def _witness(arcs, q, aw, bw, state, value, mode):
    chosen = []
    path = state.path
    while path is not None:
        path, vertex = path
        chosen.append(vertex % q)
    right = tuple(sorted(chosen))
    marks = [0] * q
    for vertex in right:
        marks[vertex] = 1
    prefix = [0]
    for value_at in marks + marks:
        prefix.append(prefix[-1] + value_at)
    left = []
    for row, arc in enumerate(arcs):
        first_gap = (arc.start + arc.length) % q
        length = q - arc.length
        if prefix[first_gap + length] == prefix[first_gap]:
            left.append(row)
    actual = sum(aw[row] for row in left) * sum(bw[col] for col in right)
    if actual != value:
        raise AssertionError(("gap DP witness mismatch", value, actual, right, left))
    return Result(actual, tuple(left), right, right[0] if right else None, mode)


def exact(arcs, q, left_weights=None, right_weights=None):
    """Exact gap DP; O(p+q^2*(1+sum(bw))*log(q+1)) operations.

    For unweighted B, this is O(p+q^3*log(q+1)), even for arbitrary
    binary-encoded A weights. With weighted B the bound is
    pseudopolynomial. For unit A and huge B weights, the initial
    minimum-union solver in solver.py provides a polynomial alternative.
    """
    arcs, aw, bw = validate(arcs, q, left_weights, right_weights)
    if not q or not arcs or not sum(aw) or not sum(bw):
        return Result(0, (), (), None, "gap-exact")
    universal, matrix = gap_matrix(arcs, q, aw)
    best_value, best_state = 0, None
    for anchor in range(q):
        initial = State(anchor, 0, bw[anchor], (None, anchor))
        trees = {initial.mass: Envelope(matrix, anchor, anchor + q - 1)}
        trees[initial.mass].insert(initial)
        value = initial.mass * (universal + matrix[anchor][anchor + q])
        if value > best_value:
            best_value, best_state = value, initial
        for j in range(anchor + 1, anchor + q):
            candidates = []
            # All queries precede all insertions for this j, including
            # when bw[j%q] == 0: no selected vertex can be reused.
            for resource, tree in trees.items():
                previous = tree.query(j)
                candidate = State(j, previous.score + matrix[previous.last][j], resource + bw[j % q], (previous.path, j))
                candidates.append(candidate)
                value = candidate.mass * (universal + candidate.score + matrix[j][anchor + q])
                if value > best_value:
                    best_value, best_state = value, candidate
            for candidate in candidates:
                if candidate.mass not in trees:
                    trees[candidate.mass] = Envelope(matrix, anchor, anchor + q - 1)
                trees[candidate.mass].insert(candidate)
    if best_state is None:
        return Result(0, (), (), None, "gap-exact")
    return _witness(arcs, q, aw, bw, best_state, best_value, "gap-exact")


def approximate(arcs, q, left_weights=None, right_weights=None, epsilon=Fraction(1, 10)):
    """FPTAS via budget buckets and single-crossing envelopes.

    For anchor b we only select vertices whose B weight is at most bw[b].
    Some optimal solution is covered when b is its heaviest B vertex.
    Positive retained masses then lie in [bw[b], q*bw[b]]. Thus
    K=O((q/epsilon)*log(q+1)) buckets suffice, independently of the
    magnitudes of both weight vectors. The operation bound is
    O(p+q^3/epsilon*log(q+1)*log((q+1)/epsilon)), with polynomial
    bit lengths. Memory is O(p+q^2/epsilon*log(q+1)).

    When choosing a predecessor from a bucket, its mass can lose a
    factor 1+epsilon/(2q), while its completed-gap A-score is at least
    that of the compared history. Trimming at the new endpoint costs
    at most another such factor. Induction along at most q selected
    vertices gives a final objective at least (1-epsilon)*OPT.
    """
    arcs, aw, bw = validate(arcs, q, left_weights, right_weights)
    epsilon = Fraction(epsilon)
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between zero and one")
    if not q or not arcs or not sum(aw) or not sum(bw):
        return Result(0, (), (), None, "gap-fptas")
    universal, matrix = gap_matrix(arcs, q, aw)
    best_value, best_state = 0, None
    for anchor in range(q):
        cap = bw[anchor]
        if cap == 0:
            continue
        thresholds = [cap]
        denominator = epsilon.denominator * 2 * q
        while thresholds[-1] <= q * cap:
            lo = thresholds[-1]
            thresholds.append(lo + lo * epsilon.numerator // denominator + 1)
        bucket = lambda value: bisect_right(thresholds, value) - 1
        initial = State(anchor, 0, bw[anchor], (None, anchor))
        trees = {bucket(initial.mass): Envelope(matrix, anchor, anchor + q - 1)}
        trees[bucket(initial.mass)].insert(initial)
        value = initial.mass * (universal + matrix[anchor][anchor + q])
        if value > best_value:
            best_value, best_state = value, initial
        for j in range(anchor + 1, anchor + q):
            if bw[j % q] > cap:
                continue
            candidates = {}
            for tree in trees.values():
                previous = tree.query(j)
                candidate = State(j, previous.score + matrix[previous.last][j], previous.mass + bw[j % q], (previous.path, j))
                key = bucket(candidate.mass)
                old = candidates.get(key)
                if old is None or (candidate.score, candidate.mass) > (old.score, old.mass):
                    candidates[key] = candidate
            for key, candidate in candidates.items():
                value = candidate.mass * (universal + candidate.score + matrix[j][anchor + q])
                if value > best_value:
                    best_value, best_state = value, candidate
                if key not in trees:
                    trees[key] = Envelope(matrix, anchor, anchor + q - 1)
                trees[key].insert(candidate)
    if best_state is None:
        return Result(0, (), (), None, "gap-fptas")
    return _witness(arcs, q, aw, bw, best_state, best_value, "gap-fptas")
