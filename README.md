# Circular-convex bicliques

Python implementations of exact and approximation algorithms for bicliques in
circular-convex bipartite graphs, with reproducible verification.

## Requirements

Python 3.10 or newer; standard library only.

## Input and objective

The circular part is `B = range(q)`. Each row of the other part has an
`Arc(start, length)` neighbourhood, interpreted modulo `q`. Length zero gives an
empty neighbourhood; length `q` gives a universal neighbourhood. Duplicate rows
are supported.

For a biclique `(X, Y)`, the objective is

```python
sum(left_weights[a] for a in X) * sum(right_weights[b] for b in Y)
```

Omitted weight vectors default to all ones. Weights must be nonnegative integers.
The model uses separable edge weights, and does not support arbitrary independent
edge weights, negative weights or representation recognition.

## Usage

```python
from fractions import Fraction
from solver import Arc
from fast_solver import exact, approximate

arcs = [Arc(2, 5), Arc(4, 5), Arc(0, 5)]
result = exact(arcs, 6)
assert result.objective == 9
assert result.left == (0, 1, 2)
assert result.right == (0, 2, 4)

weighted = approximate(
    arcs, 6,
    left_weights=[100, 200, 300],
    right_weights=[1, 2, 3, 4, 5, 6],
    epsilon=Fraction(1, 10),
)
print(weighted.objective, weighted.left, weighted.right)
```

Results contain an objective value and actual zero-based row and column indices.
Empty solutions have value zero. Use rational `epsilon` strictly between zero
and one; a float is converted exactly by `Fraction`.

## Implementations

- `fast_solver.py`: gap matrix, maximum-query structure, exact solver and FPTAS.
- `monge_solver.py`: unit-B layered DP using SMAWK on full rectangles.
- `solver.py`: shared classes, validation, brute-force oracle and an earlier
  minimum-union baseline. Import `approximate` from `fast_solver` for the
  weight-independent operation bound.
- `verify_fast.py`: exhaustive, randomized and witness verification.
- `fast_verification.json`: output from the recorded verification run.
- `verify_monge.py`, `monge_verification.json`: separate oracle and transition
  checks of the rectangular-search solver, including a staircase-padding
  counterexample.
- `SHA256.json`: digests of the code, documentation and recorded report.

Write `p` for the number of rows, `q` for the circular part size and `M` for the
sum of its weights. Arithmetic-operation bounds for the compact arc input are:

- Exact: `O(p + q^2 * (M+1) * log(q+1))`, which is pseudopolynomial for weighted B.
- With unit B weights: `O(p + q^3 * log(q+1))`, allowing binary weights on A.
  The same bound applies to `monge_solver.exact_unit_b(arcs, q, left_weights)`.
  This is an alternative implementation, not an asymptotic improvement over
  the envelope solver.
- Approximation: `O(p + q^3/epsilon * log(q+1) * log((q+1)/epsilon))`, independently
  of numerical weight magnitudes. The value is at least `(1-epsilon) * OPT` and
  integer operands have polynomial bit length.
- With unit A weights, `solver.solve(..., mode="profit")` provides the
  complementary exact `O(q*p^3)` minimum-union baseline.

The baseline follows the interval Minimum k-Union approach; see
[Chlamtac et al.](https://doi.org/10.4230/LIPIcs.APPROX-RANDOM.2016.6).
Single-crossing maximum queries and geometric state reduction are standard tools.

The rectangular solver applies [SMAWK](https://doi.org/10.1007/BF01840359)
only to full matrices. The invalid part of the predecessor staircase cannot
simply be padded by minus infinity while preserving total monotonicity. For
`q=5`, the rows `Arc(2,4), Arc(4,3), Arc(4,3)` give layer-3 best predecessors
`1,2,1` at endpoints `2,3,4` for anchor `0`, even with all weights equal to one.
More advanced staircase primitives yield a theoretical deterministic
`O(p+q^3*ack(q))` bound (Klawe-Kleitman) or a Las Vegas expected `O(p+q^3)`
bound (Chan), where `ack` is inverse Ackermann. Those primitives are not
implemented here; these stronger bounds do not describe the included code.

```python
from monge_solver import exact_unit_b
assert exact_unit_b(arcs, 6).objective == 9
```

## Reproduce verification

```bash
python3 verify_fast.py
python3 verify_monge.py
```

Run with assertions enabled, without `-O`. Each driver overwrites its report
(`fast_verification.json` or `monge_verification.json`); elapsed times vary.

The checks cover 77,495 exhaustive multisets of up to four neighbourhood arcs for
`q <= 6`, 2,500 small weighted random instances, 1,200 further instances with
weights between `10^11` and `10^12-1`, and 96,420 maximum queries. Reconstructed
witnesses are checked directly against the original neighbourhood predicate.

The exact oracle enumerates subsets of A and intersects original neighbourhoods;
it does not use the gap recurrence. No exact mismatches or approximation-bound
violations occurred. One approximate result was 1476 against optimum 1480 at
`epsilon=1/4`; the complete instance is in the JSON report. Finite checks do not
replace mathematical proofs.

The rectangular solver is separately checked on the same 77,495 exhaustive
families, 2,500 random instances with arbitrary A weights (including large
binary weights), and 600 layers with arbitrary predecessor scores (6,828
endpoint comparisons). No exact discrepancies or infeasible witnesses were
found. These runs do not test the unimplemented advanced staircase primitives.
