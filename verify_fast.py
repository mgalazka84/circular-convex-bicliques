"""Independent checks of the gap matrix, envelope tree, exact DP and FPTAS."""

from fractions import Fraction
from itertools import combinations_with_replacement
from pathlib import Path
import json
import random
import time

from solver import Arc, all_arcs, brute_force
from fast_solver import State, Envelope, gap_matrix, exact, approximate


def check_witness(result, arcs, q, aw=None, bw=None):
    aw = [1] * len(arcs) if aw is None else aw
    bw = [1] * q if bw is None else bw
    assert len(set(result.left)) == len(result.left)
    assert len(set(result.right)) == len(result.right)
    assert all(arcs[a].contains(b, q) for a in result.left for b in result.right)
    assert result.objective == sum(aw[a] for a in result.left) * sum(bw[b] for b in result.right)


def main():
    started = time.perf_counter()
    report = {"seeds": {"weighted": 20261019, "envelope": 20261020, "large_weights": 20261021}, "exhaustive": []}
    # Exhaustive neighbourhood families up to row permutation. Independent
    # oracle enumerates all subsets of A and intersects the original rows.
    for q in range(1, 7):
        count = 0
        for p in range(5):
            for family in combinations_with_replacement(all_arcs(q), p):
                oracle = brute_force(family, q)
                result = exact(family, q)
                assert oracle.objective == result.objective, (q, family, oracle, result)
                check_witness(result, family, q)
                count += 1
        report["exhaustive"].append({"q": q, "maximum_p": 4, "families": count})
        print("Exhaustive", q, count, flush=True)

    rng = random.Random(20261019)
    nonoptimal = 0
    controlled = None
    for case in range(2500):
        q = rng.randrange(1, 13)
        p = rng.randrange(1, 11)
        arcs = [Arc(rng.randrange(q), rng.randrange(q + 1)) for _ in range(p)]
        aw = [rng.randrange(21) for _ in arcs]
        bw = [rng.randrange(21) for _ in range(q)]
        oracle = brute_force(arcs, q, aw, bw)
        got = exact(arcs, q, aw, bw)
        assert got.objective == oracle.objective, (case, arcs, aw, bw, oracle, got)
        check_witness(got, arcs, q, aw, bw)
        epsilon = [Fraction(1, 10), Fraction(1, 4), Fraction(1, 2)][case % 3]
        approx = approximate(arcs, q, aw, bw, epsilon)
        assert approx.objective <= oracle.objective
        assert approx.objective * epsilon.denominator >= (epsilon.denominator - epsilon.numerator) * oracle.objective
        check_witness(approx, arcs, q, aw, bw)
        nonoptimal += approx.objective < oracle.objective
        if controlled is None and approx.objective < oracle.objective:
            controlled = {"arcs": [[a.start, a.length] for a in arcs], "q": q, "A_weights": aw, "B_weights": bw, "epsilon": str(epsilon), "exact_value": oracle.objective, "approximate_value": approx.objective, "ratio": str(Fraction(approx.objective, oracle.objective))}
    report["weighted_small"] = {"cases": 2500, "maximum_p": 10, "maximum_q": 12, "weight_range": [0, 20], "exact_mismatches": 0, "approximation_violations": 0, "nonoptimal_approximate_outputs": nonoptimal}
    print("Small weighted complete", flush=True)

    rng = random.Random(20261020)
    queries = 0
    for case in range(200):
        q = rng.randrange(1, 16)
        arcs = [Arc(rng.randrange(q), rng.randrange(q + 1)) for _ in range(20)]
        aw = [rng.randrange(11) for _ in arcs]
        _, matrix = gap_matrix(arcs, q, aw)
        # Verify every legitimate circular gap directly from the original
        # neighbourhoods, without using interval endpoints or prefix sums.
        for i in range(q):
            for j in range(i + 1, i + q + 1):
                open_gap = {(x % q) for x in range(i + 1, j)}
                expected = sum(weight for arc, weight in zip(arcs, aw) if arc.length not in (0, q) and set(range(q)) - {x for x in range(q) if arc.contains(x, q)} <= open_gap)
                assert matrix[i][j] == expected
        tree = Envelope(matrix, 0, 2 * q - 1)
        functions = []
        for insertion in range(30):
            state = State(rng.randrange(2 * q), rng.randrange(-30, 31), rng.randrange(101), None)
            functions.append(state)
            tree.insert(state)
            for x in range(2 * q):
                got = tree.query(x)
                key = lambda f: (f.score + matrix[f.last][x], f.mass)
                assert key(got) == max(map(key, functions))
                queries += 1
    report["envelope"] = {"families": 200, "functions_per_family": 30, "point_queries_after_insertions": queries, "mismatches": 0, "all_circular_gap_values_checked_directly": True}
    print("Envelope complete", queries, flush=True)

    rng = random.Random(20261021)
    large_nonoptimal = 0
    for case in range(1200):
        q = rng.randrange(1, 11)
        p = rng.randrange(1, 11)
        arcs = [Arc(rng.randrange(q), rng.randrange(q + 1)) for _ in range(p)]
        aw = [rng.randrange(10**11, 10**12) for _ in arcs]
        bw = [rng.randrange(10**11, 10**12) for _ in range(q)]
        epsilon = [Fraction(1, 10), Fraction(1, 4), Fraction(1, 2)][case % 3]
        oracle = brute_force(arcs, q, aw, bw)
        got = approximate(arcs, q, aw, bw, epsilon)
        assert got.objective <= oracle.objective
        assert got.objective * epsilon.denominator >= (epsilon.denominator - epsilon.numerator) * oracle.objective
        check_witness(got, arcs, q, aw, bw)
        large_nonoptimal += got.objective < oracle.objective
        if controlled is None and got.objective < oracle.objective:
            controlled = {"arcs": [[a.start, a.length] for a in arcs], "q": q, "A_weights": aw, "B_weights": bw, "epsilon": str(epsilon), "exact_value": oracle.objective, "approximate_value": got.objective, "ratio": str(Fraction(got.objective, oracle.objective))}
    report["large_binary_weights"] = {"cases": 1200, "maximum_p": 10, "maximum_q": 10, "weight_range": [10**11, 10**12 - 1], "epsilon_values": ["1/10", "1/4", "1/2"], "approximation_violations": 0, "nonoptimal_approximate_outputs": large_nonoptimal}
    report["first_nonoptimal_example"] = controlled
    for q, arcs in [(0, []), (0, [Arc(0, 0)]), (1, []), (1, [Arc(0, 1)])]:
        assert exact(arcs, q).objective == brute_force(arcs, q).objective
        assert approximate(arcs, q).objective == brute_force(arcs, q).objective
    report["boundary_cases"] = 4
    report["infeasible_witnesses"] = 0
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    Path("fast_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
