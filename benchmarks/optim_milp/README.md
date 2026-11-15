# intlinprog benchmarks

Deterministic MILP families (no random numbers), used to measure the effect of each part of
`intlinprog`: node counts and times.

| Script | Problems |
|---|---|
| `bench_milp9.m` | 9 MILPs: general integers (M11, M13, G1-G4) and multidimensional 0/1 knapsacks (K1-K3) |
| `bench_fl('basic')` | 3 capacitated facility-location problems with big-M rows (presolve and MIR cuts matter) |
| `bench_many(rule)` | 36 problems of the three families; shifted geometric means of nodes and time |
| `bench_hard(rule)` | 12 harder general-integer problems (hundreds to tens of thousands of nodes) |

Run them in MatLivre from this folder, e.g. `matlivre-cli bench_milp9.m`, or with a script that
calls `bench_hard('maxfun')`. They also run in MATLAB, so the node counts can be compared with
MATLAB's own `intlinprog` (whose default is HiGHS since R2024a).
