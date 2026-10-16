"""Independent oracle and transition tests for the rectangular Monge solver."""

from itertools import combinations_with_replacement
from pathlib import Path
import json
import random
import time

from solver import Arc, all_arcs, brute_force
from fast_solver import gap_matrix
from monge_solver import exact_unit_b, layer_maxima, row_maxima
from verify_fast import check_witness


def main():
    started = time.perf_counter()
    report = {"algorithm": "deterministic rectangular SMAWK, O(p+q^3 log(q+1))",
              "advanced_staircase_primitives_implemented": False,
              "seeds": {"weighted_A": 317041, "layers": 317042}, "exhaustive": []}
    for q in range(1, 7):
        count = 0
        for p in range(5):
            for family in combinations_with_replacement(all_arcs(q), p):
                oracle = brute_force(family, q)
                got = exact_unit_b(family, q)
                assert got.objective == oracle.objective, (q, family, got, oracle)
                check_witness(got, family, q)
                count += 1
        report["exhaustive"].append({"q": q, "maximum_p": 4, "families": count})
        print("Monge exhaustive", q, count, flush=True)

    rng = random.Random(317041)
    for case in range(2500):
        q = rng.randrange(1, 16)
        arcs = [Arc(rng.randrange(q), rng.randrange(q + 1)) for _ in range(rng.randrange(1, 12))]
        aw = [rng.randrange(10**12) if case % 2 else rng.randrange(6) for _ in arcs]
        oracle = brute_force(arcs, q, aw)
        got = exact_unit_b(arcs, q, aw)
        assert got.objective == oracle.objective, (case, arcs, aw, got, oracle)
        check_witness(got, arcs, q, aw)
    report["weighted_A"] = {"cases": 2500, "maximum_p": 11, "maximum_q": 15,
                            "exact_mismatches": 0, "infeasible_witnesses": 0}

    rng = random.Random(317042)
    comparisons = 0
    for case in range(600):
        q = rng.randrange(2, 25)
        arcs = [Arc(rng.randrange(q), rng.randrange(q + 1)) for _ in range(25)]
        aw = [rng.randrange(6) for _ in arcs]
        _, matrix = gap_matrix(arcs, q, aw)
        anchor = rng.randrange(q)
        previous = {i: rng.randrange(-100, 101) for i in range(anchor, anchor + q) if rng.randrange(3)}
        got = layer_maxima(matrix, previous, anchor, q)
        expected = {}
        for j in range(anchor, anchor + q):
            candidates = [(score + matrix[i][j], i) for i, score in previous.items() if i < j]
            if candidates:
                expected[j] = max(candidates)
                comparisons += 1
        assert got == expected, (case, got, expected)
    report["arbitrary_score_layers"] = {"cases": 600, "endpoint_comparisons": comparisons, "mismatches": 0}

    # Three unit-weight rows: forbidden intervals [1,1], [2,3], [2,3].
    # The layer-3 predecessor maxima are 1,2,1 at endpoints 2,3,4.
    # Padding the invalid entries with -infinity is not totally monotone.
    q = 5
    arcs = [Arc(2, 4), Arc(4, 3), Arc(4, 3)]
    _, matrix = gap_matrix(arcs, q, [1, 1, 1])
    previous = {i: matrix[0][i] for i in range(1, q)}
    layer = layer_maxima(matrix, previous, 0, q)
    assert [layer[j][1] for j in (2, 3, 4)] == [1, 2, 1]
    assert [[previous[i] + matrix[i][j] if i < j else None for i in (1, 2, 3)] for j in (2, 3, 4)] == [[0, None, None], [0, 1, None], [2, 1, 1]]
    report["padding_counterexample"] = {"q": q, "arcs": [[a.start, a.length] for a in arcs],
                                        "anchor": 0, "layer": 3, "predecessor_maxima": [1, 2, 1]}
    # A tie plateau must choose the first column in the supplied order.
    assert row_maxima(list(range(15)), [8, 6, 3, 0], lambda r, c: 0) == {r: 8 for r in range(15)}
    assert row_maxima([], [1], lambda r, c: 0) == {}
    for q, arcs in [(0, []), (0, [Arc(0, 0)]), (1, []), (1, [Arc(0, 1)])]:
        assert exact_unit_b(arcs, q).objective == brute_force(arcs, q).objective
    report["boundary_cases"] = 4
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    Path("monge_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
