"""Unit-B gap DP using SMAWK only on fully defined rectangles.

The staircase is split recursively into rectangles. This deterministic,
practical implementation takes O(p + q^3 log(q+1)) arithmetic operations.
It does NOT implement the more advanced Klawe-Kleitman or randomized Chan
staircase primitives. In particular, it makes no cubic worst-case claim.

SMAWK: Aggarwal et al., Algorithmica 2 (1987), 195-208,
https://doi.org/10.1007/BF01840359.
"""

from fast_solver import State, gap_matrix, _witness
from solver import Result, validate


def row_maxima(rows, columns, value):
    """Leftmost row maxima of an implicit totally monotone FULL matrix.

    Orders are the supplied sequence orders, not numeric index order.
    O(len(rows) + len(columns)) value evaluations and auxiliary space.
    The caller must establish total monotonicity; padding arbitrary
    staircase blanks by minus infinity is not a valid justification.
    """
    if not rows or not columns:
        return {}
    reduced = []
    for column in columns:
        while reduced:
            row = rows[len(reduced) - 1]
            if value(row, column) <= value(row, reduced[-1]):
                break
            reduced.pop()
        if len(reduced) < len(rows):
            reduced.append(column)
    answer = row_maxima(rows[1::2], reduced, value)
    positions = {column: k for k, column in enumerate(reduced)}
    start = 0
    for index in range(0, len(rows), 2):
        row = rows[index]
        stop = positions[answer[rows[index + 1]]] if index + 1 < len(rows) else len(reduced) - 1
        best = start
        for k in range(start + 1, stop + 1):
            if value(row, reduced[k]) > value(row, reduced[best]):
                best = k
        answer[row] = reduced[best]
        start = stop
    return answer


def layer_maxima(matrix, previous, anchor, q):
    """Return (score, predecessor) for all endpoints in one exact layer.

    previous maps reachable predecessors to their completed gap scores.
    Earlier predecessor functions gain relative to later ones as the
    endpoint increases. Hence increasing endpoints and DECREASING
    predecessors form a full totally monotone matrix for maximization.
    Splitting at a midpoint covers every valid i<j pair exactly once.
    All rectangles at a recursion depth have O(q) total dimensions;
    a layer uses O(q log(q+1)) evaluations. Ties prefer larger i.
    """
    answer = {}

    def split(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        columns = [i for i in range(mid - 1, lo - 1, -1) if i in previous]
        if columns:
            rows = list(range(mid, hi))
            value = lambda j, i: previous[i] + matrix[i][j]
            winners = row_maxima(rows, columns, value)
            for j, i in winners.items():
                candidate = (value(j, i), i)
                if j not in answer or candidate > answer[j]:
                    answer[j] = candidate
        split(lo, mid)
        split(mid, hi)

    split(anchor, anchor + q)
    return answer


def exact_unit_b(arcs, q, left_weights=None):
    """Exact unit-B solver, with arbitrary nonnegative integer A weights.

    Time O(p + q^3 log(q+1)); space O(p + q^2), including witness paths.
    Uses rectangular SMAWK decomposition, not a randomized primitive.
    """
    arcs, aw, bw = validate(arcs, q, left_weights, None)
    mode = "gap-monge-rectangles"
    if not q or not arcs or not sum(aw):
        return Result(0, (), (), None, mode)
    universal, matrix = gap_matrix(arcs, q, aw)
    best_value, best_state = 0, None
    for anchor in range(q):
        initial = State(anchor, 0, 1, (None, anchor))
        previous_states = {anchor: initial}
        value = universal + matrix[anchor][anchor + q]
        if value > best_value:
            best_value, best_state = value, initial
        for count in range(2, q + 1):
            previous = {i: state.score for i, state in previous_states.items()}
            winners = layer_maxima(matrix, previous, anchor, q)
            current = {}
            for j, (score, i) in winners.items():
                state = State(j, score, count, (previous_states[i].path, j))
                current[j] = state
                value = count * (universal + score + matrix[j][anchor + q])
                if value > best_value:
                    best_value, best_state = value, state
            previous_states = current
    if best_state is None:
        return Result(0, (), (), None, mode)
    return _witness(arcs, q, aw, bw, best_state, best_value, mode)
