# MatLivre toolbox vs MATLAB R2026b

MATLAB R2026b Update 1 (Optimization and Global Optimization Toolboxes) against the MatLivre
toolbox (v6, plus the fixes listed at the end), on the 10 test files of `matlab_reference/`.
Files: `reference_matlab.txt` (MATLAB) and `reference_matlivre.txt` (MatLivre).

Note: `AGENTS.md` asks for expected outputs from MATLAB R2023b. R2026b differs from R2023b in
`linprog` and `intlinprog`, which have used HiGHS since R2024a; the `CutGeneration` option no longer
exists there. The PR still needs an R2023b run.

## Second run (fixed runner, v7): counts

All 10 test files ran in MATLAB without failures. Of the 71 deterministic cases
(`RESULT`/`HS`/`MILP` lines; `compare_reference.py` prints them side by side):

- **50** with the same optimal value (relative difference < 1e-5) and the same exitflag;
- **6** with the same optimal value and a different exitflag (HS100 1/2, N4 3/2, N5 3/2, S2 1/2,
  ps_linear 1/2, M11 1/2);
- **6** with the same status and no value to compare (infeasible/unbounded LP and MILP,
  ga success counts);
- **7** with a different value: P7 (another local minimum; MATLAB better), HS13 (MatLivre better),
  U7 (unbounded in both; only the value at the stop differs), ga_nonlcon_hybrid (MATLAB slightly
  infeasible, 0.9987 < 1), ga_mixedint_eq (MATLAB 35.62, wrong; MatLivre exact), ps_nonlcon
  (MATLAB stops at -1.9975; MatLivre exact), M8 (HiGHS presolve solves the knapsack at the root,
  so MaxNodes = 1 does not stop it);
- **2** with a different status: infeasible fmincon (MATLAB 0, MatLivre -2) and nonconvex QP
  (MATLAB 2, MatLivre -6).

The 60 random LPs: MATLAB 60/60 and MatLivre 60/60 against HiGHS. On the hard MILP (M11), HiGHS
closes the gap in 1997 nodes and 0.4 s; MatLivre needs about 22000 nodes and 12 s.

## Summary (first run)

| Area | Result |
|---|---|
| `fmincon` (9 cases + 9 HS) | Same optima everywhere except P7: MATLAB found the global minimum (f = 1.1566), MatLivre another valid local minimum (f = 1.8951, at the active constraint nearest x0). On HS13 (degenerate), MatLivre gets f = 0.99994, MATLAB 1.3506. MatLivre uses fewer iterations (P1: 13 vs 24). Infeasible problem: MATLAB exitflag 0, MatLivre -2. |
| `linprog` | Same optimal values and the same duals (LP2: `[0 1.5 1]`). LP6 has several optimal vertices; MATLAB/HiGHS and MatLivre return different ones. 60 random LPs: MATLAB and MatLivre both agree 60/60 with HiGHS. |
| `quadprog`, `lsqlin` | Same solutions and multipliers. Nonconvex QP: MATLAB R2026b returns exitflag 2 (a local solution), MatLivre -6 (refuses, as R2023b's interior-point-convex did). |
| `fminunc` | **After the fix, identical to MATLAB**: same iterations and function counts on 5 of 6 problems (e.g. Rosenbrock 36/138, Beale 15/48, Powell 33/170, ill-conditioned 30/403) and errors of the same size. |
| `lsqnonlin`, `lsqcurvefit`, `fsolve` | Same solutions. Exit flags sometimes 2 where MATLAB gives 3 (step vs function-change criterion). |
| `particleswarm` | Rastrigin 2D 19/20 (MATLAB 20/20), 5D 5/20 (MATLAB 3/20), Ackley 10D 18/20 (MATLAB 20/20). |
| `ga` | **After the fix**: Rastrigin 2D 20/20 (MATLAB 12/20), 5D 16/20 (MATLAB 11/20), Ackley 10D 18/20 (MATLAB 20/20). Nonlinear constraint 20/20 (MATLAB 16/20). Mixed integer with a linear equality: MatLivre exact, MATLAB f = 35.6 (optimum 0.215). |
| `patternsearch` | Same on Rosenbrock and on linear constraints. Nonlinear constraint: MatLivre exact (ef 1), MATLAB f = -1.9975 (ef 0). |
| `MultiStart`, `GlobalSearch` | Same global minimum and the same number of distinct minima (5 and 4). GlobalSearch: 18 local runs against 13 in MATLAB. |
| `checkGradients` | **After the fix**: same verdicts and the same relative errors (wrong entry: 1 in both; correct gradients: 5e-8 vs 6.3e-8). |
| OutputFcn / PlotFcn | Same behaviour (stop gives -1, `init`/`iter`/`done`). `fsolve` stop: MATLAB counts 2 iterations, MatLivre 3. |
| `intlinprog` | Same optimal values (M1–M7). When the optimum is not unique, the point can differ. |

## Fixes made after the comparison

1. **`fminunc`**: the first-order tolerance is now relative to the initial gradient
   (`||g||inf <= tol*max(1, ||g0||inf)`), as MATLAB's results show, and the stop on a small change
   in f only fires near machine precision. Before: exitflags 2 and 3 where MATLAB gives 1.
   After: same exitflags, iterations and function counts.
2. **`checkGradients`**: forward differences by default (`FiniteDifferenceType`), and the relative
   difference is divided by the supplied derivative, `|supplied - fd| ./ max(|supplied|, 1)`. Now
   the numbers match MATLAB.
3. **`ga`**: the mutation scale goes from twice the population spread down to once the spread over
   the generations (it was once the spread). Rastrigin 5D went from 1/20 to 16/20.
4. **Test files**: `test_checkgradients` no longer uses `deal` in the objective (MATLAB's
   `checkGradients` calls it with one output); `test_milp` sets `CutGeneration` only where the
   option exists.
5. **`record_reference`** is now a function: before, the test scripts overwrote the runner's
   variables (`t`, `i`), which caused the spurious "Argument to TOC" failures and wrong labels.

## Still different (by design or for later)

- `fmincon` P7: a different local minimum. Both are valid; this follows from SQP and x0.
- Exit flags of `lsqnonlin`/`fsolve` (2 vs 3) and the infeasible `fmincon` case (-2 vs 0).
- Nonconvex `quadprog`: a local solver for nonconvex QPs (as in R2026b) is not implemented.
- `particleswarm` on Ackley 10D (18/20 vs 20/20).
