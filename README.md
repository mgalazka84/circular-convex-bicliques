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
- `solver.py`: shared classes, validation, brute-force oracle and an earlier
  minimum-union baseline. Import `approximate` from `fast_solver` for the
  weight-independent operation bound.
- `verify_fast.py`: exhaustive, randomized and witness verification.
- `fast_verification.json`: output from the recorded verification run.
- `SHA256.json`: digests of the code, documentation and recorded report.

Write `p` for the number of rows, `q` for the circular part size and `M` for the
sum of its weights. Arithmetic-operation bounds for the compact arc input are:

- Exact: `O(p + q^2 * (M+1) * log(q+1))`, which is pseudopolynomial for weighted B.
- With unit B weights: `O(p + q^3 * log(q+1))`, allowing binary weights on A.
- Approximation: `O(p + q^3/epsilon * log(q+1) * log((q+1)/epsilon))`, independently
  of numerical weight magnitudes. The value is at least `(1-epsilon) * OPT` and
  integer operands have polynomial bit length.
- With unit A weights, `solver.solve(..., mode="profit")` provides the
  complementary exact `O(q*p^3)` minimum-union baseline.

The baseline follows the interval Minimum k-Union approach; see
[Chlamtac et al.](https://doi.org/10.4230/LIPIcs.APPROX-RANDOM.2016.6).
Single-crossing maximum queries and geometric state reduction are standard tools.

## Reproduce verification

```bash
python3 verify_fast.py
```

Run with assertions enabled, without `-O`. The driver overwrites
`fast_verification.json`, whose elapsed time may differ from the recorded run.

The checks cover 77,495 exhaustive multisets of up to four neighbourhood arcs for
`q <= 6`, 2,500 small weighted random instances, 1,200 further instances with
weights between `10^11` and `10^12-1`, and 96,420 maximum queries. Reconstructed
witnesses are checked directly against the original neighbourhood predicate.

The exact oracle enumerates subsets of A and intersects original neighbourhoods;
it does not use the gap recurrence. No exact mismatches or approximation-bound
violations occurred. One approximate result was 1476 against optimum 1480 at
`epsilon=1/4`; the complete instance is in the JSON report. Finite checks do not
replace mathematical proofs.
