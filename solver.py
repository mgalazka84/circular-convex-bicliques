"""Exact research baseline for circular-convex bipartite bicliques.

Rows are vertices of A.  Each neighbourhood in the cyclically ordered set
B = range(m) is given as Arc(start, length).  A length of zero represents
an isolated A vertex; a length of m represents a universal A vertex.

The objective is (sum of A weights) * (sum of B weights).  Without weights,
this is the number of edges.  Arbitrary EDGE weights are not supported.

The cut-and-complement reduction and a left-to-right union DP are implemented
here.  Its relation to Minimum p-Union in interval hypergraphs must be kept
explicit when assessing novelty.  This is a research baseline, not a claim
of publication priority.
"""

from dataclasses import dataclass
from bisect import bisect_right
from fractions import Fraction
from numbers import Integral


@dataclass(frozen=True)
class Arc:
    start: int
    length: int

    def contains(self, vertex, m):
        return (vertex - self.start) % m < self.length

    def mask(self, m):
        return sum(1 << ((self.start + i) % m) for i in range(self.length))


@dataclass(frozen=True)
class Result:
    objective: int
    left: tuple
    right: tuple
    anchor: int | None
    mode: str


def validate(arcs, m, left_weights=None, right_weights=None):
    if not isinstance(m, Integral) or m < 0:
        raise ValueError("m must be a nonnegative integer")
    arcs = tuple(arcs)
    for arc in arcs:
        if not isinstance(arc, Arc):
            raise TypeError("neighbourhoods must be Arc instances")
        if not isinstance(arc.start, Integral) or not isinstance(arc.length, Integral):
            raise ValueError("arc parameters must be integers")
        if not 0 <= arc.length <= m or not 0 <= arc.start < max(m, 1):
            raise ValueError("invalid circular arc")
    aw = tuple(1 for _ in arcs) if left_weights is None else tuple(left_weights)
    bw = tuple(1 for _ in range(m)) if right_weights is None else tuple(right_weights)
    if len(aw) != len(arcs) or len(bw) != m:
        raise ValueError("weight vector has the wrong length")
    if any(not isinstance(w, Integral) or w < 0 for w in aw + bw):
        raise ValueError("weights must be nonnegative integers")
    return arcs, aw, bw


def anchor_intervals(arcs, m, anchor, left_weights):
    """Return empty-gap rows, and (left, right, row id, profit) intervals.

    Coordinates 0,...,m-2 correspond to anchor+1,...,anchor+m-1 modulo m.
    Rows not adjacent to the anchor cannot belong to this anchored biclique.
    """
    universal = []
    intervals = []
    for row, (arc, profit) in enumerate(zip(arcs, left_weights)):
        if not arc.contains(anchor, m):
            continue
        if arc.length == m:
            universal.append(row)
            continue
        first_gap = (arc.start + arc.length) % m
        lo = (first_gap - anchor - 1) % m
        hi = lo + m - arc.length - 1
        if hi >= m - 1:
            raise AssertionError("a forbidden interval crossed the anchor")
        intervals.append((lo, hi, row, profit))
    intervals.sort()
    return tuple(universal), intervals


def _prefix(values):
    answer = [0]
    for value in values:
        answer.append(answer[-1] + value)
    return answer


def _anchored(arcs, m, anchor, aw, bw, mode, trace=False, doubled_prefix=None):
    universal, intervals = anchor_intervals(arcs, m, anchor, aw)
    free_profit = sum(aw[row] for row in universal)
    prefix = _prefix(bw + bw) if doubled_prefix is None else doubled_prefix
    total_b = prefix[m]
    base = anchor + 1
    # State = (rightmost selected endpoint, union cost / selected profit).
    # Endpoint -1 denotes an empty family.  Values contain the other quantity.
    states = {(-1, 0): (0, None)}
    for lo, hi, row, profit in intervals:
        updated = dict(states)
        for (rightmost, resource), (value, path) in states.items():
            new_endpoint = max(rightmost, hi)
            added_cost = prefix[base + hi + 1] - prefix[base + max(lo, rightmost + 1)] if hi > rightmost else 0
            if mode == "profit":
                key = (new_endpoint, resource + profit)
                candidate = value + added_cost
                better = key not in updated or candidate < updated[key][0]
            else:
                key = (new_endpoint, resource + added_cost)
                candidate = value + profit
                better = key not in updated or candidate > updated[key][0]
            if better:
                updated[key] = (candidate, (path, row) if trace else None)
        states = updated
    best = (0, None)
    for (rightmost, resource), (value, path) in states.items():
        if mode == "profit":
            selected_profit, union_cost = free_profit + resource, value
        else:
            selected_profit, union_cost = free_profit + value, resource
        objective = selected_profit * (total_b - union_cost)
        if objective > best[0]:
            best = (objective, path)
    if not trace:
        return best[0], None
    selected = list(universal)
    path = best[1]
    while path is not None:
        path, row = path
        selected.append(row)
    return best[0], tuple(sorted(selected))


def solve(arcs, m, left_weights=None, right_weights=None, mode="auto"):
    """Return an optimal biclique and its objective.

    Profit DP: O(m * p^2 * (1 + sum(A weights))) arithmetic operations.
    Cost DP:   O(m * p^2 * (1 + sum(B weights))) arithmetic operations.
    p = len(arcs).  With unweighted A the first bound is O(m*p^3),
    even when B weights are binary-encoded and large.  With unweighted B
    the second is O(p^2*m^2), even when A weights are large.
    For two binary-weighted sides these are pseudopolynomial bounds.

    The value pass retains only two DP layers.  A second pass for the winning
    anchor stores persistent predecessor nodes to reconstruct a witness.
    """
    arcs, aw, bw = validate(arcs, m, left_weights, right_weights)
    if mode == "auto":
        mode = "profit" if sum(aw) <= sum(bw) else "cost"
    if mode not in ("profit", "cost"):
        raise ValueError("mode must be auto, profit, or cost")
    if not m or not arcs or not sum(aw) or not sum(bw):
        return Result(0, (), (), None, mode)
    best_value, best_anchor = 0, None
    doubled_prefix = _prefix(bw + bw)
    for anchor in range(m):
        value, _ = _anchored(arcs, m, anchor, aw, bw, mode, doubled_prefix=doubled_prefix)
        if value > best_value:
            best_value, best_anchor = value, anchor
    if best_anchor is None:
        return Result(0, (), (), None, mode)
    value, left = _anchored(arcs, m, best_anchor, aw, bw, mode, trace=True, doubled_prefix=doubled_prefix)
    common = (1 << m) - 1
    for row in left:
        common &= arcs[row].mask(m)
    right = tuple(i for i in range(m) if common >> i & 1)
    actual = sum(aw[row] for row in left) * sum(bw[col] for col in right)
    if actual != value or value != best_value:
        raise AssertionError("reconstructed witness and DP value disagree")
    return Result(actual, left, right, best_anchor, mode)


def brute_force(arcs, m, left_weights=None, right_weights=None):
    """Independent oracle: enumerate all A subsets and intersect neighbourhoods."""
    arcs, aw, bw = validate(arcs, m, left_weights, right_weights)
    masks = [arc.mask(m) for arc in arcs]
    best = Result(0, (), (), None, "brute-force")
    for chosen in range(1, 1 << len(arcs)):
        common = (1 << m) - 1
        left = []
        for row, mask in enumerate(masks):
            if chosen >> row & 1:
                common &= mask
                left.append(row)
        right = tuple(i for i in range(m) if common >> i & 1)
        value = sum(aw[row] for row in left) * sum(bw[col] for col in right)
        if value > best.objective:
            best = Result(value, tuple(left), right, None, "brute-force")
    return best


def _profit_thresholds(total, epsilon, p):
    """Exact integer buckets with upper/lower ratio at most 1+epsilon/p.

    A bucket starts at L and ends at L+floor(epsilon*L/p), inclusive.
    This avoids numerical logarithms and huge rational powers.  For small
    profits the buckets are singletons. Profit zero has its own bucket.
    """
    thresholds = [1]
    numerator, denominator = epsilon.numerator, epsilon.denominator * p
    while thresholds[-1] <= total:
        lo = thresholds[-1]
        thresholds.append(lo + lo * numerator // denominator + 1)
    return thresholds


def _anchored_trimmed(arcs, m, anchor, aw, bw, thresholds, trace=False, doubled_prefix=None):
    universal, intervals = anchor_intervals(arcs, m, anchor, aw)
    free_profit = sum(aw[row] for row in universal)
    prefix = _prefix(bw + bw) if doubled_prefix is None else doubled_prefix
    total_b = prefix[m]
    base = anchor + 1
    bucket = lambda value: bisect_right(thresholds, value) - 1
    # Within each endpoint/profit bucket, retain minimum exact union cost.
    # Value tuple = (actual profit, actual cost, persistent predecessor).
    states = {(-1, bucket(free_profit)): (free_profit, 0, None)}
    for lo, hi, row, profit in intervals:
        updated = dict(states)
        for (rightmost, _), (old_profit, old_cost, path) in states.items():
            new_endpoint = max(rightmost, hi)
            added_cost = prefix[base + hi + 1] - prefix[base + max(lo, rightmost + 1)] if hi > rightmost else 0
            new_profit, new_cost = old_profit + profit, old_cost + added_cost
            key = (new_endpoint, bucket(new_profit))
            old = updated.get(key)
            if old is None or (new_cost, -new_profit) < (old[1], -old[0]):
                updated[key] = (new_profit, new_cost, (path, row) if trace else None)
        states = updated
    best = (0, None)
    for profit, cost, path in states.values():
        objective = profit * (total_b - cost)
        if objective > best[0]:
            best = (objective, path)
    if not trace:
        return best[0], None
    left = list(universal)
    path = best[1]
    while path is not None:
        path, row = path
        left.append(row)
    return best[0], tuple(sorted(left))


def approximate(arcs, m, left_weights=None, right_weights=None, epsilon=Fraction(1, 10)):
    """Return a feasible biclique of value at least (1-epsilon)*OPT.

    Both vertex-weight vectors may contain binary-encoded nonnegative
    integers. An existing edge has weight aw[a]*bw[b]. This does not handle
    arbitrary independently specified edge weights.

    For p A-vertices, U=sum(aw), the number K of buckets is
    O((p/epsilon)*(1+log(1+U))). There are O(p*K) states per layer,
    giving O(m*p^2*K*log(K)) arithmetic operations. Integer operands have
    polynomial bit length, so this is an FPTAS in the standard bit model.

    Each trim preserves cost and loses at most a 1+epsilon/p factor in
    profit. After at most p trims the loss is at most (1+epsilon/p)^p,
    whose reciprocal is at least 1-epsilon. See the research note.
    """
    arcs, aw, bw = validate(arcs, m, left_weights, right_weights)
    epsilon = Fraction(epsilon)
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be strictly between zero and one")
    if not m or not arcs or not sum(aw) or not sum(bw):
        return Result(0, (), (), None, "fptas")
    thresholds = _profit_thresholds(sum(aw), epsilon, len(arcs))
    doubled_prefix = _prefix(bw + bw)
    best_value, best_anchor = 0, None
    for anchor in range(m):
        value, _ = _anchored_trimmed(arcs, m, anchor, aw, bw, thresholds, doubled_prefix=doubled_prefix)
        if value > best_value:
            best_value, best_anchor = value, anchor
    if best_anchor is None:
        return Result(0, (), (), None, "fptas")
    value, left = _anchored_trimmed(arcs, m, best_anchor, aw, bw, thresholds, trace=True, doubled_prefix=doubled_prefix)
    common = (1 << m) - 1
    for row in left:
        common &= arcs[row].mask(m)
    right = tuple(i for i in range(m) if common >> i & 1)
    actual = sum(aw[row] for row in left) * sum(bw[col] for col in right)
    if actual != value or value != best_value:
        raise AssertionError("approximate witness and DP value disagree")
    return Result(actual, left, right, best_anchor, "fptas")


def all_arcs(m):
    """One canonical encoding of each cyclic interval, including empty/full."""
    if m == 0:
        return [Arc(0, 0)]
    return [Arc(0, 0), Arc(0, m)] + [Arc(start, length) for length in range(1, m) for start in range(m)]
