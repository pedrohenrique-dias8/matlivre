# MatLivre optimization toolbox: draft

Nine solvers from MATLAB's Optimization Toolbox and six components of the Global Optimization
Toolbox, written in the MATLAB language so that they run
in MatLivre today. They could later move to Rust builtins. This is a clean-room implementation:
the algorithms come from J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.
(Springer, 2006), and from the papers cited in each file, and the interfaces from MATLAB's
public documentation. No MathWorks or Octave source code was used.

## Use

```matlab
addpath('toolbox/optim')          % from the repository root
x = fmincon(@(x) sum((x - 1).^2), [0 0], [1 1], 1)
```

| Folder | Contents |
|---|---|
| `toolbox/optim/` | Solvers, helpers and plot functions |
| `toolbox/optim/tests/` | Test scripts (`test_*.m`) and the DC circuit example from issue #8 |
| `toolbox/optim/matlab_reference/` | `record_reference.m` (runs the tests with MATLAB's own solvers), the recorded outputs and `COMPARISON.md` |
| `toolbox/optim/python_reference/` | SciPy/HiGHS references and `compare_reference.py` |

To run a test in MatLivre: `cd toolbox/optim/tests`, then `addpath('..')` and `test_fmincon`.

## Solvers

| Function | Problem | Method |
|---|---|---|
| `fmincon` | Nonlinear objective; linear, nonlinear and bound constraints | SQP: damped BFGS, elastic QP, l1 merit function, second-order correction |
| `fminunc` | Unconstrained nonlinear | BFGS on the inverse Hessian, strong-Wolfe line search with cubic zoom |
| `quadprog` | Convex QP | Primal-dual interior point (Mehrotra) |
| `linprog` | LP | `dual-simplex` (default, as in MATLAB): bounded dual simplex, vertex solutions, duals from the basis; `interior-point`: the QP solver with H = 0 |
| `lsqlin` | Constrained linear least squares | Convex QP with C'C |
| `intlinprog` | Mixed-integer LP | Branch and bound on the dual simplex, each node warm-started from the parent's basis; Gomory mixed-integer cuts at the root (`CutGeneration`); depth-first until the first incumbent, then best bound; branching `maxfun` / `reliability` (strong branching + pseudocosts) / `maxpscost` / `mostfractional`; rounding heuristic; pruning by integrality of the objective; gaps, x0, MaxNodes/MaxTime, OutputFcn |
| `lsqnonlin` | Nonlinear least squares with bounds | Levenberg–Marquardt (Marquardt scaling, Nielsen damping, active set on the bounds) |
| `lsqcurvefit` | Curve fitting `fun(x, xdata) ≈ ydata` | Through `lsqnonlin` |
| `fsolve` | Nonlinear systems F(x) = 0 | Trust-region dogleg |
| `optimoptions` | Options | Struct; solvers also accept optimset-style names |
| **Global Optimization Toolbox** | | |
| `particleswarm` | Bound-constrained global | PSO with adaptive inertia and random neighbourhoods, `HybridFcn`, vectorized evaluation |
| `ga` | Global with linear, nonlinear, bound and integer constraints | Rank scaling, stochastic universal selection, elitism, scattered and intermediate crossover, Gaussian mutation adapted to the population spread; Deb's feasibility rules; projection onto linear equalities |
| `patternsearch` | Derivative-free local, with constraints | GPS 2N, with extra poll directions along nearly active linear constraints; feasibility rules |
| `MultiStart` | Many local minima | classdef with `run(ms, problem, k)` or explicit start points; merges solutions by XTolerance/FunctionTolerance |
| `GlobalSearch` | Global minimum with `fmincon` | Trial points, stage 1 and 2, basins of attraction and a score threshold |
| `createOptimProblem` | Problem struct | For fmincon, fminunc, lsqnonlin, lsqcurvefit |
| **Tools** | | |
| `checkGradients` | Check user derivatives | Central differences; gradients, Jacobians and constraint gradients (`IsConstraint`); `valid` and the relative differences |
| `OutputFcn` / `PlotFcn` | Monitor and stop any solver | In every solver except the QP/LP ones (whose MATLAB algorithms do not offer them either); exitflag -1 on stop. MATLAB's call conventions: `(x, optimValues, state)` for local solvers, `(optimValues, state)` for particleswarm, `(options, state, flag)` for ga, `(optimvalues, options, flag)` for patternsearch |
| Plot functions | | `optimplotfval`, `optimplotx`, `optimplotfunccount`, `optimplotconstrviolation`, `optimplotstepsize`, `optimplotfirstorderopt`, `optimplotresnorm`, `pswplotbestf`, `gaplotbestf`, `psplotbestf`, `psplotmeshsize` |

Shared helpers: `optim_simplex` (dense bounded dual simplex with warm start: an artificial box
makes every basis dual feasible by bound flipping, so there is no phase 1), `optim_outfcn` (OutputFcn/PlotFcn calls, one figure with a subplot per plot
function), `optim_plotscalar`, `optim_hybrid` (`HybridFcn`), `optim_qpip` (dense convex QP: interior point with KKT equilibration and
adaptive regularization), `optim_qpsolve` (common front end of the QP-based solvers),
`optim_fdjac` (finite-difference Jacobian that respects bounds), `optim_getopt` and
`optim_display`.

Every solver accepts the MATLAB calling forms: positional arguments, `problem` struct, every
output, user derivatives through `SpecifyObjectiveGradient` / `SpecifyConstraintGradient`,
forward or central differences, and matrix-shaped `x0`. Exit flags follow MATLAB's meanings:
1, 2 and 3 for success; 0 when a limit is reached; -2 infeasible or no root; -3 unbounded;
-6 nonconvex.

## Validation (MatLivre 0.1.8)

Each test file has a SciPy counterpart (`ref_*.py`). Every case agrees with SciPy, or with the
published optimum, to 7 to 12 digits unless noted below.

- `test_fmincon.m`: 9 cases (bounds, linear, nonlinear, user gradients, central differences,
  matrix x0, problem struct, infeasible problem → -2).
- `test_hs.m`: Hock–Schittkowski 6, 7, 13, 26, 39, 40, 43, 65 and 100. HS13 (no constraint
  qualification) stops at f = 1.0006 against f* = 1.
- `test_qp.m`: 6 LPs (including infeasible → -2, unbounded → -3, and a 30×20 problem), 4 QPs
  (including nonconvex → -6) and 2 lsqlin problems. With the default dual simplex the LP solutions are vertices (with `interior-point`, a point inside the optimal face when the optimum is not unique). The objective matches HiGHS.
- `test_fminunc.m`: Rosenbrock in 2D and 10D, Beale, Powell singular, an ill-conditioned problem
  with matrix x0, and an unbounded problem → -3. Accuracy matches SciPy BFGS.
- `test_global.m`: success rates over 20 seeds. Rastrigin 2D: PSO 19/20, GA 18/20.
  Rastrigin 5D: PSO 5/20, GA 1/20. Ackley 10D: PSO 18/20, GA 20/20. GA with a nonlinear
  constraint 20/20, integer GA 20/20, mixed-integer GA with a linear equality at the exact
  optimum, hybrid fmincon. patternsearch on Rosenbrock, linear and nonlinear constraints.
  MultiStart finds both global minima of the six-hump camel and 3 other local minima;
  GlobalSearch and MultiStart work with lsqcurvefit.
- `test_simplex.m`: 60 random LPs (35 optimal, 5 infeasible, 20 unbounded, with free and
  one-sided variables and equalities), checked by `ref_simplex.py` against HiGHS: 60/60 agree on
  status and optimal value.
- `test_milp.m`: 0/1 knapsacks (12 and 30 items, the second strongly correlated), mixed problem
  with equality, 5×5 assignment, set covering, general integers with negative bounds, integer
  infeasibility → -2, unbounded relaxation → -3, MaxNodes with x0 → 2, problem struct. Optimal
  values match SciPy/HiGHS in every case; where the optimum is not unique, the point can differ.
  A 15-variable general-integer problem with a weak relaxation finds the optimum but needs about
  22000 nodes (12 s with the simplex; 70 s with the earlier interior-point nodes) to close the
  gap, against 2242 for HiGHS. The test stops it at 2000 nodes. On 12 random general-integer
  problems the root cuts cut the total from 6674 to 2704 nodes (3.6 s to 2.1 s); M13 is one of
  them (5633 to 1831 nodes).
- `test_checkgradients.m`: correct and wrong gradients, Jacobians, constraint gradients (a
  wrong entry is detected at the right element), matrix x0.
- `test_outputfcn.m`: history recording (`init`/`iter`/`done`), stopping after 3 iterations in
  fmincon, fminunc, lsqnonlin and fsolve (exitflag -1), every plot function without changing
  the result, and stopping and plot functions for particleswarm, ga and patternsearch.
- `test_nls.m`: least squares with bounds (with multipliers), exponential and Lorentzian fits,
  and Brown–Dennis (large residual: f* is reached, but it takes 1360 function evaluations
  against 400 in MINPACK). fsolve: 2×2, Powell badly scaled, 2×2 matrix cube root, a system
  with no real root → -2, and Broyden tridiagonal with Jacobian.

## Known limits / next steps

1. Expected outputs recorded from MATLAB R2023b, as `AGENTS.md` requires (the current
   reference is R2026b), and tests under `crates/livre-lang/tests/all/`. Displays and messages do not yet use MATLAB's exact wording.
2. Only one algorithm per solver (`Algorithm` is ignored). No `HessianFcn`. Everything is
   dense. Plot functions draw in MatLivre's figure model; in the headless `matlivre-cli` they
   run but nothing is shown.
3. `lsqnonlin` is slow on large-residual problems. A Moré-style trust region would be the fix.
4. `lsqnonlin` and `lsqcurvefit` in R2023b also accept linear and nonlinear constraints; not
   done yet.
5. `intlinprog` has no presolve, and only Gomory cuts, only at the root (no MIR, cover,
   clique or flow cuts, no probing, no conflict analysis). This is the main gap with MATLAB and
   HiGHS on hard MILPs. The dual simplex is dense and refactorizes the basis at every iteration
   (fine up to a few hundred rows).
6. Still missing: `fminimax`, `fgoalattain`, `fseminf`, `coneprog`, and the
   problem-based workflow (`optimproblem`, `optimvar`, `solve`). From the Global
   Optimization Toolbox: `simulannealbnd`, `surrogateopt`, `gamultiobj`, `paretosearch`,
   `CustomStartPointSet` and `RandomStartPointSet`.
7. Constraint handling in `ga` and `patternsearch` (Deb's rules) differs from MATLAB's
   (augmented Lagrangian or penalty). GlobalSearch draws uniform random trial points instead of
   using scatter search. Both are documented in the help text of each file.
8. MatLivre bugs found along the way, all with an issue drafted: local functions colliding
   across files (every helper here has a per-file prefix to work around it), local functions
   of a classdef file not visible to its methods (helpers here are static private methods),
   `warning` state, and the JIT falling back without notice. Also seen: `mfilename('fullpath')`
   returns an empty string in `matlivre-cli`, so the tests fall back to `pwd` to find the toolbox
   (run them from `tests/`).

## Comparison with MATLAB

`matlab_reference/` holds `record_reference.m`, which runs the scripts in `tests/` with
MATLAB's own solvers and writes `reference_matlab.txt`. `COMPARISON.md` has the results. `reference_matlivre.txt` holds the same
tests run in MatLivre, in the same line format, ready for comparison.

**Release.** The reference was recorded with MATLAB R2026b, the only release available to the
author; `AGENTS.md` asks for R2023b. Known difference between the two: `linprog` and
`intlinprog` use HiGHS by default since R2024a. The legacy algorithms are still available in
R2026b (`'dual-simplex-legacy'`, `'legacy'`) and would give an R2023b-like reference; the
current file uses the defaults. No change to the default algorithm is known for `fmincon`,
`fminunc`, `fsolve`, `lsqlin`, `lsqnonlin`, `lsqcurvefit`, `quadprog`, `ga`, `particleswarm`,
`patternsearch`, `MultiStart` and `GlobalSearch`, so R2026b should be a valid reference for
them.
