function [x, fval, exitflag, output, population, scores] = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, IntCon, options)
%GA  Global minimization by a genetic algorithm.
%   x = ga(fun, nvars)
%   x = ga(fun, nvars, A, b)
%   x = ga(fun, nvars, A, b, Aeq, beq)
%   x = ga(fun, nvars, A, b, Aeq, beq, lb, ub)
%   x = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon)
%   x = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, options)
%   x = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, IntCon)
%   x = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, IntCon, options)
%   x = ga(problem)
%   [x, fval, exitflag, output, population, scores] = ga(...)
%
%   fun takes a row vector of length nvars (or, with UseVectorized, a matrix with one individual
%   per row). IntCon lists the variables that must take integer values.
%
%   Algorithm: rank fitness scaling, stochastic universal selection, elitism (EliteCount),
%   scattered crossover (CrossoverFraction of the non-elite children), Gaussian mutation whose
%   scale shrinks with the generations, children clamped to the bounds and integer variables
%   rounded. Linear and nonlinear constraints use Deb's feasibility rules: a feasible individual
%   is better than an infeasible one, two infeasible individuals are compared by their total
%   constraint violation. This constraint handling differs from MATLAB's ga (augmented Lagrangian
%   or penalty), so results on constrained problems will not match MATLAB's run by run.
%
%   Options (optimoptions('ga', ...)): PopulationSize (50 if nvars <= 5, else 200),
%   MaxGenerations (100*nvars), MaxStallGenerations (50), FunctionTolerance (1e-6),
%   ConstraintTolerance (1e-3), FitnessLimit (-Inf), MaxTime (Inf), EliteCount
%   (ceil(0.05*PopulationSize)), CrossoverFraction (0.8), InitialPopulationMatrix ([]),
%   InitialPopulationRange ([-10; 10]), UseVectorized (false), HybridFcn ([] | 'fmincon' |
%   'fminunc' | 'fminsearch' | 'patternsearch'), Display ('final' | 'iter' | 'off').
%   exitflag: 1 average relative change in the best value over MaxStallGenerations below
%   FunctionTolerance (constraints satisfied), 5 FitnessLimit reached, 0 MaxGenerations,
%   -2 no feasible point, -5 MaxTime exceeded.
%
%   Reference: D. E. Goldberg (1989); J. E. Baker (1987), stochastic universal sampling;
%   K. Deb (2000), "An efficient constraint handling method for genetic algorithms".
%   Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(fun)
    p = fun; fun = p.fitnessfcn; nvars = p.nvars;
    A = ga_fld(p, 'Aineq'); b = ga_fld(p, 'Bineq'); Aeq = ga_fld(p, 'Aeq'); beq = ga_fld(p, 'Beq');
    lb = ga_fld(p, 'lb'); ub = ga_fld(p, 'ub'); nonlcon = ga_fld(p, 'nonlcon');
    IntCon = ga_fld(p, 'intcon'); options = ga_fld(p, 'options'); nin = 11;
else
    nin = nargin;
end
if nin < 3, A = []; end
if nin < 4, b = []; end
if nin < 5, Aeq = []; end
if nin < 6, beq = []; end
if nin < 7, lb = []; end
if nin < 8, ub = []; end
if nin < 9, nonlcon = []; end
if nin < 10, IntCon = []; end
if nin < 11, options = struct(); end
if nin == 10 && isstruct(IntCon), options = IntCon; IntCon = []; end   % ga(..., nonlcon, options)
if isempty(options), options = struct(); end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
if ischar(nonlcon) || isstring(nonlcon), nonlcon = str2func(nonlcon); end
n = nvars;
if isempty(A), A = zeros(0, n); b = zeros(0, 1); end
if isempty(Aeq), Aeq = zeros(0, n); beq = zeros(0, 1); end
b = b(:); beq = beq(:);
if isempty(lb), lb = -inf(1, n); end
if isempty(ub), ub = inf(1, n); end
lb = double(lb(:)).'; ub = double(ub(:)).';
IntCon = IntCon(:).';
lb(IntCon) = ceil(lb(IntCon)); ub(IntCon) = floor(ub(IntCon));

if n <= 5, defpop = 50; else, defpop = 200; end
P = optim_getopt(options, {'PopulationSize'}, defpop);
maxgen = optim_getopt(options, {'MaxGenerations', 'Generations'}, 100 * n);
maxstall = optim_getopt(options, {'MaxStallGenerations', 'StallGenLimit'}, 50);
tolf = optim_getopt(options, {'FunctionTolerance', 'TolFun'}, 1e-6);
tolc = optim_getopt(options, {'ConstraintTolerance', 'TolCon'}, 1e-3);
flim = optim_getopt(options, {'FitnessLimit'}, -inf);
maxtime = optim_getopt(options, {'MaxTime', 'TimeLimit'}, inf);
ne = optim_getopt(options, {'EliteCount'}, ceil(0.05 * P));
cf = optim_getopt(options, {'CrossoverFraction'}, 0.8);
X0 = optim_getopt(options, {'InitialPopulationMatrix', 'InitialPopulation'}, []);
ipr = optim_getopt(options, {'InitialPopulationRange', 'PopInitRange'}, [-10; 10]);
vect = optim_getopt(options, {'UseVectorized', 'Vectorized'}, false);
if ischar(vect) || isstring(vect), vect = strcmpi(vect, 'on'); end
hyb = optim_getopt(options, {'HybridFcn'}, []);
lvl = optim_display(options, 'final');

output = struct('problemtype', '', 'rngstate', rng, 'generations', 0, 'funccount', 0, ...
    'message', '', 'maxconstraint', 0, 'hybridflag', []);
if any(lb > ub)
    x = []; fval = []; exitflag = -2; population = []; scores = [];
    output.message = 'No feasible point: lb(i) > ub(i) for some i (after rounding for integer variables).';
    if lvl ~= 0, fprintf('\n%s\n\n', output.message); end
    return
end
hascon = ~isempty(A) || ~isempty(Aeq) || ~isempty(nonlcon);
if ~isempty(IntCon), output.problemtype = 'integerconstraints';
elseif ~isempty(nonlcon), output.problemtype = 'nonlinearconstr';
elseif hascon, output.problemtype = 'linearconstraints';
elseif any(isfinite([lb ub])), output.problemtype = 'boundconstraints';
else, output.problemtype = 'unconstrained'; end
tic0 = tic;

% ---- initial population inside the bounds (InitialPopulationRange where a bound is infinite)
lo = ipr(1, :) .* ones(1, n); hi = ipr(end, :) .* ones(1, n);
lo(isfinite(lb)) = lb(isfinite(lb)); hi(isfinite(ub)) = ub(isfinite(ub));
lo = min(lo, hi); hi = max(hi, lo);
pop = lo + rand(P, n) .* (hi - lo);
k0 = min(size(X0, 1), P);
if k0 > 0, pop(1:k0, :) = X0(1:k0, :); end
pop = ga_fix(pop, lb, ub, IntCon, Aeq, beq);
[F, V] = ga_eval(fun, nonlcon, pop, A, b, Aeq, beq, vect); nev = P;
rangev = hi - lo; rangev(rangev == 0) = 1;
[best, bv, bx] = ga_best(F, V, pop, tolc);
hist = best * ones(maxstall + 1, 1);
if lvl == 2
    fprintf('\n                                  Best           Mean      Stall\n');
    fprintf('Generation      Func-count        f(x)           f(x)    Generations\n');
end
exitflag = 0; msg = sprintf('Optimization terminated: maximum number of generations (%d) exceeded.', maxgen);
stallgen = 0; g = 0;
OF = optim_outfcn('setup', options, 'ga');
gast = struct('Generation', 0, 'StartTime', tic0, 'StopFlag', [], 'Best', best, 'LastImprovement', 0, ...
    'Population', pop, 'Score', F, 'FunEval', nev, 'Expectation', [], 'Selection', []);
userstop = false;
if OF.active
    [s1, OF, gast] = optim_outfcn(OF, options, gast, 'init');
    [s2, OF, gast] = optim_outfcn(OF, options, gast, 'iter');
    userstop = s1 || s2;
    if userstop, maxgen = 0; end
end
nx = round(cf * (P - ne));            % crossover children
nm = P - ne - nx;                     % mutation children
for g = 1:maxgen
    % ---- fitness scaling by rank (Deb ordering), expectation proportional to 1/sqrt(rank)
    key = ga_rankkey(F, V, tolc);
    [~, ord] = sort(key);
    expct = zeros(P, 1); expct(ord) = 1 ./ sqrt(1:P);
    expct = expct * (2 * (nx) + nm) / sum(expct);
    % ---- stochastic universal sampling
    npar = 2 * nx + nm;
    parents = ga_sus(expct, npar);
    parents = parents(randperm(npar));
    % ---- elite
    kids = zeros(P, n);
    kids(1:ne, :) = pop(ord(1:ne), :);
    % ---- crossover: half scattered (gene exchange), half intermediate (random blend per gene)
    for i = 1:nx
        p1 = pop(parents(2 * i - 1), :); p2 = pop(parents(2 * i), :);
        if mod(i, 2) == 1
            mask = rand(1, n) < 0.5;
            c = p2; c(mask) = p1(mask);
        else
            c = p1 + rand(1, n) .* (p2 - p1);
        end
        kids(ne + i, :) = c;
    end
    % ---- Gaussian mutation: scale twice the spread of the population, falling to once the spread
    %      at MaxGenerations, with a decaying floor to keep exploring
    spread = std(pop, 0, 1);
    sc = max((2 - g / maxgen) * spread, (1 - g / maxgen) * 0.02 * rangev);
    for i = 1:nm
        pm = pop(parents(2 * nx + i), :);
        kids(ne + nx + i, :) = pm + sc .* randn(1, n);
    end
    pop = ga_fix(kids, lb, ub, IntCon, Aeq, beq);
    [F, V] = ga_eval(fun, nonlcon, pop, A, b, Aeq, beq, vect); nev = nev + P;
    [bestg, bvg, bxg] = ga_best(F, V, pop, tolc);
    if ga_better(bestg, bvg, best, bv, tolc)
        best = bestg; bv = bvg; bx = bxg; stallgen = 0;
    else
        stallgen = stallgen + 1;
    end
    hist = [hist(2:end); best];
    if lvl == 2
        fin = isfinite(F);
        fprintf('%7d %15d %15.6g %15.6g %8d\n', g, nev, best, mean(F(fin)), stallgen);
    end
    if OF.active
        gast.Generation = g; gast.Best(end + 1) = best; gast.Population = pop; gast.Score = F;
        gast.FunEval = nev; gast.Expectation = expct; gast.Selection = parents;
        if stallgen == 0, gast.LastImprovement = g; end
        [userstop, OF, gast] = optim_outfcn(OF, options, gast, 'iter');
        if userstop, break, end
    end
    if bv <= tolc && best <= flim
        exitflag = 5; msg = 'Optimization terminated: minimum fitness limit reached.'; break
    end
    if g > maxstall && bv <= tolc && mean(abs(diff(hist))) / max(1, abs(best)) <= tolf
        exitflag = 1; msg = 'Optimization terminated: average change in the fitness value less than FunctionTolerance.'; break
    end
    if toc(tic0) > maxtime
        exitflag = -5; msg = 'Optimization terminated: MaxTime exceeded.'; break
    end
end
x = bx; fval = best;
if bv > tolc
    exitflag = -2; msg = 'Optimization terminated: no feasible point found (constraint violation above ConstraintTolerance).';
end
if userstop
    exitflag = -1; msg = 'Optimization terminated: stop requested by an output function or plot function.';
end
if OF.active, optim_outfcn(OF, options, gast, 'done'); end
if ~isempty(hyb) && isempty(IntCon) && bv <= tolc
    [x, fval, hflag, hev] = optim_hybrid(hyb, fun, x, fval, lb, ub, ga_emptyz(A), ga_emptyz(b), ga_emptyz(Aeq), ga_emptyz(beq), nonlcon);
    output.hybridflag = hflag; nev = nev + hev;
end
output.generations = g; output.funccount = nev; output.message = msg; output.maxconstraint = bv;
population = pop; scores = F;
if lvl == 1 || lvl == 2 || (lvl == -1 && exitflag <= 0)
    fprintf('\n%s\n\n', msg);
end
end

% ====================================================================== helpers

function X = ga_fix(X, lb, ub, IntCon, Aeq, beq)
% clamp to the bounds, round integer variables, and project the continuous variables onto
% the linear equalities (alternating with the bounds a few times)
X = min(max(X, lb), ub);
if ~isempty(IntCon)
    X(:, IntCon) = round(X(:, IntCon));
    X = min(max(X, lb), ub);
end
if nargin >= 5 && ~isempty(Aeq)
    n = size(X, 2);
    cont = true(1, n); cont(IntCon) = false;
    Ac = Aeq(:, cont);
    if any(Ac(:))
        Pc = pinv(Ac);
        for rep = 1:5
            r = X * Aeq.' - beq.';                  % residual per individual (rows)
            if max(abs(r(:))) < 1e-12, break, end
            X(:, cont) = X(:, cont) - r * Pc.';
            X = min(max(X, lb), ub);
        end
    end
end
end

function [F, V] = ga_eval(fun, nonlcon, X, A, b, Aeq, beq, vect)
m = size(X, 1);
if vect
    F = double(fun(X)); F = F(:);
else
    F = zeros(m, 1);
    for i = 1:m, F(i) = fun(X(i, :)); end
end
F(isnan(F)) = inf;
V = zeros(m, 1);
if ~isempty(A), V = V + sum(max(0, X * A.' - b.'), 2); end
if ~isempty(Aeq), V = V + sum(abs(X * Aeq.' - beq.'), 2); end
if ~isempty(nonlcon)
    for i = 1:m
        [c, ceq] = nonlcon(X(i, :));
        V(i) = V(i) + sum(max(0, c(:))) + sum(abs(ceq(:)));
    end
end
end

function key = ga_rankkey(F, V, tolc)
% Deb's rules as a sortable key: feasible by f, then infeasible by violation
feas = V <= tolc;
key = zeros(size(F));
if any(feas), fmax = max(F(feas)); else, fmax = 0; end
key(feas) = F(feas);
key(~feas) = fmax + abs(fmax) + 1 + V(~feas) * 1e6;
key(~isfinite(key)) = realmax;
end

function [best, bv, bx] = ga_best(F, V, X, tolc)
key = ga_rankkey(F, V, tolc);
[~, i] = min(key);
best = F(i); bv = V(i); bx = X(i, :);
end

function tf = ga_better(f1, v1, f2, v2, tolc)
if v1 <= tolc && v2 <= tolc, tf = f1 < f2;
elseif v1 <= tolc, tf = true;
elseif v2 <= tolc, tf = false;
else, tf = v1 < v2; end
end

function idx = ga_sus(expct, k)
% stochastic universal sampling of k parents with expectations expct (sum = k)
c = cumsum(expct(:));
c = c / c(end) * k;
pts = rand + (0:k - 1).';
idx = zeros(k, 1); j = 1;
for i = 1:k
    while j < numel(c) && c(j) < pts(i), j = j + 1; end
    idx(i) = j;
end
end

function v = ga_emptyz(v)
if isempty(v), v = []; end
end

function v = ga_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end
