function [x, fval, exitflag, output, points] = particleswarm(fun, nvars, lb, ub, options)
%PARTICLESWARM  Bound-constrained global minimization by particle swarm optimization.
%   x = particleswarm(fun, nvars)
%   x = particleswarm(fun, nvars, lb, ub)
%   x = particleswarm(fun, nvars, lb, ub, options)
%   x = particleswarm(problem)
%   [x, fval, exitflag, output, points] = particleswarm(...)
%
%   fun takes a row vector of length nvars (or, with UseVectorized, a matrix with one particle
%   per row, returning a column of values).
%
%   Algorithm (following the public description of MATLAB's particleswarm): random initial swarm
%   inside the bounds (or InitialSwarmSpan when a bound is infinite); each particle moves under
%   inertia plus attraction to its own best position and to the best position among a random
%   subset of the other particles. The neighbourhood grows while the swarm stalls and resets when
%   the global best improves; the inertia adapts within InertiaRange. Positions are clamped to the
%   bounds and the clamped velocity components are zeroed.
%
%   Options (optimoptions('particleswarm', ...)): SwarmSize (min(100, 10*nvars)),
%   MaxIterations (200*nvars), MaxStallIterations (20), FunctionTolerance (1e-6),
%   ObjectiveLimit (-Inf), MaxTime (Inf), InertiaRange ([0.1 1.1]), SelfAdjustmentWeight (1.49),
%   SocialAdjustmentWeight (1.49), MinNeighborsFraction (0.25), InitialSwarmMatrix ([]),
%   InitialSwarmSpan (2000), UseVectorized (false), HybridFcn ([] | 'fmincon' | 'fminunc' |
%   'fminsearch' | 'patternsearch', or {fcn, options}), Display ('final' | 'iter' | 'off').
%   exitflag: 1 relative change in the best value below FunctionTolerance over
%   MaxStallIterations, 0 MaxIterations, -2 lb > ub, -3 ObjectiveLimit reached, -5 MaxTime.
%
%   Reference: J. Kennedy and R. Eberhart (1995); M. E. H. Pedersen (2010), "Good parameters for
%   particle swarm optimization". Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(fun)
    p = fun; fun = p.objective; nvars = p.nvars;
    lb = pso_fld(p, 'lb'); ub = pso_fld(p, 'ub'); options = pso_fld(p, 'options');
    nin = 5;
else
    nin = nargin;
end
if nin < 3, lb = []; end
if nin < 4, ub = []; end
if nin < 5 || isempty(options), options = struct(); end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
n = nvars;
if isempty(lb), lb = -inf(1, n); end
if isempty(ub), ub = inf(1, n); end
lb = double(lb(:)).'; ub = double(ub(:)).';
S = optim_getopt(options, {'SwarmSize'}, min(100, 10 * n));
maxit = optim_getopt(options, {'MaxIterations', 'MaxIter'}, 200 * n);
maxstall = optim_getopt(options, {'MaxStallIterations'}, 20);
tolf = optim_getopt(options, {'FunctionTolerance', 'TolFun'}, 1e-6);
flim = optim_getopt(options, {'ObjectiveLimit'}, -inf);
maxtime = optim_getopt(options, {'MaxTime'}, inf);
wr = optim_getopt(options, {'InertiaRange'}, [0.1 1.1]);
c1 = optim_getopt(options, {'SelfAdjustmentWeight'}, 1.49);
c2 = optim_getopt(options, {'SocialAdjustmentWeight'}, 1.49);
mnf = optim_getopt(options, {'MinNeighborsFraction'}, 0.25);
X0 = optim_getopt(options, {'InitialSwarmMatrix', 'InitialSwarm'}, []);
span = optim_getopt(options, {'InitialSwarmSpan'}, 2000);
vect = optim_getopt(options, {'UseVectorized', 'Vectorized'}, false);
if ischar(vect) || isstring(vect), vect = strcmpi(vect, 'on'); end
hyb = optim_getopt(options, {'HybridFcn'}, []);
lvl = optim_display(options, 'final');

output = struct('rngstate', rng, 'iterations', 0, 'funccount', 0, 'message', '', 'hybridflag', []);
points = struct('X', [], 'Fval', []);
if any(lb > ub)
    x = []; fval = []; exitflag = -2; output.message = 'The bounds are inconsistent: lb(i) > ub(i) for some i.';
    if lvl ~= 0, fprintf('\n%s\n\n', output.message); end
    return
end
tic0 = tic;

% ---- initial swarm and velocities
lo = lb; hi = ub;
if numel(span) == 1, span = span * ones(1, n); end
for j = 1:n
    if ~isfinite(lo(j)) && ~isfinite(hi(j)), lo(j) = -span(j) / 2; hi(j) = span(j) / 2;
    elseif ~isfinite(lo(j)), lo(j) = hi(j) - span(j);
    elseif ~isfinite(hi(j)), hi(j) = lo(j) + span(j); end
end
X = lo + rand(S, n) .* (hi - lo);
k0 = min(size(X0, 1), S);
if k0 > 0, X(1:k0, :) = X0(1:k0, :); end
X = min(max(X, lb), ub);
rngv = hi - lo;
V = rngv .* (2 * rand(S, n) - 1);
F = pso_eval(fun, X, vect); nev = S;
P = X; Pf = F;
[fbest, ib] = min(F); xbest = X(ib, :);
nmin = max(2, floor(S * mnf));
N = nmin; W = max(wr); stall = 0;
hist = fbest * ones(maxstall + 1, 1);
if lvl == 2
    fprintf('\n                                 Best            Mean     Stall\n');
    fprintf('Iteration     f-count            f(x)            f(x)    Iterations\n');
    fprintf('%5d %13d %15.6g %15.6g %8d\n', 0, nev, fbest, mean(F(isfinite(F))), 0);
end
exitflag = 0; msg = sprintf('Optimization ended: the number of iterations reached MaxIterations = %d.', maxit);
it = 0;
OF = optim_outfcn('setup', options, 'pso');
if OF.active
    [s1, OF] = optim_outfcn(OF, pso_ov(xbest, fbest, 0, nev, F, 0, X), 'init');
    [s2, OF] = optim_outfcn(OF, pso_ov(xbest, fbest, 0, nev, F, 0, X), 'iter');
    if s1 || s2, exitflag = -1; msg = 'Optimization ended: stopped by an output function or plot function.'; maxit = 0; end
end
for it = 1:maxit
    % best neighbour of each particle among N random others
    G = zeros(S, n);
    for i = 1:S
        others = randperm(S - 1, min(N, S - 1));
        others(others >= i) = others(others >= i) + 1;
        nb = [i, others];
        [~, j] = min(Pf(nb));
        G(i, :) = P(nb(j), :);
    end
    V = W * V + c1 * rand(S, n) .* (P - X) + c2 * rand(S, n) .* (G - X);
    X = X + V;
    out = X < lb | X > ub;
    X = min(max(X, lb), ub);
    V(out) = 0;
    F = pso_eval(fun, X, vect); nev = nev + S;
    imp = F < Pf;
    P(imp, :) = X(imp, :); Pf(imp) = F(imp);
    [fmin, i] = min(F);
    if fmin < fbest
        fbest = fmin; xbest = X(i, :);
        stall = max(0, stall - 1);
        N = nmin;
    else
        stall = stall + 1;
        N = min(N + nmin, S - 1);
    end
    % adaptive inertia: explore while improving, contract while stalling
    if stall < 2, W = 2 * W; elseif stall > 5, W = W / 2; end
    W = min(max(W, min(wr)), max(wr));
    hist = [hist(2:end); fbest];
    if lvl == 2
        fprintf('%5d %13d %15.6g %15.6g %8d\n', it, nev, fbest, mean(F(isfinite(F))), stall);
    end
    if OF.active
        [ustop, OF] = optim_outfcn(OF, pso_ov(xbest, fbest, it, nev, F, stall, X), 'iter');
        if ustop, exitflag = -1; msg = 'Optimization ended: stopped by an output function or plot function.'; break, end
    end
    if fbest <= flim
        exitflag = -3; msg = 'Optimization ended: the best value is below ObjectiveLimit.'; break
    end
    if it > maxstall && abs(hist(1) - hist(end)) <= tolf * max(1, abs(hist(end)))
        exitflag = 1; msg = sprintf('Optimization ended: the relative change in the best value over MaxStallIterations = %d is below FunctionTolerance.', maxstall); break
    end
    if toc(tic0) > maxtime
        exitflag = -5; msg = 'Optimization ended: MaxTime exceeded.'; break
    end
end
x = xbest; fval = fbest;
if OF.active, optim_outfcn(OF, pso_ov(xbest, fbest, it, nev, F, stall, X), 'done'); end
output.iterations = it; output.funccount = nev; output.message = msg;
points.X = P; points.Fval = Pf;
if ~isempty(hyb)
    [x, fval, hflag, hev] = optim_hybrid(hyb, fun, x, fval, lb, ub, [], [], [], [], []);
    output.hybridflag = hflag; output.funccount = output.funccount + hev;
end
if lvl == 1 || lvl == 2 || (lvl == -1 && exitflag <= 0)
    fprintf('\n%s\n\n', msg);
end
end

function F = pso_eval(fun, X, vect)
if vect
    F = double(fun(X)); F = F(:);
else
    F = zeros(size(X, 1), 1);
    for i = 1:size(X, 1), F(i) = fun(X(i, :)); end
end
F(isnan(F)) = inf;
end

function v = pso_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end

function ov = pso_ov(xb, fb, it, nev, F, stall, X)
ov = struct('bestx', xb, 'bestfval', fb, 'iteration', it, 'funccount', nev, ...
    'meanfval', mean(F(isfinite(F))), 'stalliterations', stall, 'swarm', X, 'swarmfvals', F);
end
