function [x, fval, exitflag, output, grad, hessian] = fminunc(fun, x0, options)
%FMINUNC  Find a minimum of an unconstrained multivariable function.
%   x = fminunc(fun, x0)
%   x = fminunc(fun, x0, options)
%   x = fminunc(problem)
%   [x, fval, exitflag, output, grad, hessian] = fminunc(...)
%
%   Quasi-Newton method: BFGS update of the inverse Hessian with the initial scaling of
%   Shanno and Phua, and a line search satisfying the strong Wolfe conditions (bracketing and
%   zoom with safeguarded cubic interpolation). hessian is the BFGS approximation at x.
%
%   Options (optimoptions or optimset names):
%     Display, MaxIterations/MaxIter (400), MaxFunctionEvaluations/MaxFunEvals (100*numel(x0)),
%     OptimalityTolerance/TolFun (1e-6, relative: ||g||inf <= tol*max(1, ||g0||inf)), StepTolerance/TolX (1e-6), FunctionTolerance (1e-6),
%     SpecifyObjectiveGradient/GradObj (fun returns [f, g]), FiniteDifferenceType,
%     FiniteDifferenceStepSize.
%   exitflag:  1 gradient small, 2 step small, 3 change in f small, 0 limits reached,
%             -3 objective unbounded below, -2 line search failed.
%
%   Reference: Nocedal and Wright, Numerical Optimization, 2nd ed., Alg. 3.5-3.6 and 6.1.
%   Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(fun)
    p = fun; fun = p.objective; x0 = p.x0;
    if isfield(p, 'options'), options = p.options; else, options = struct(); end
    nin = 3;
else
    nin = nargin;
end
if nin < 3 || isempty(options), options = struct(); end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
shape = size(x0); x = double(x0(:)); n = numel(x);
maxit = optim_getopt(options, {'MaxIterations', 'MaxIter'}, 400);
maxfev = optim_getopt(options, {'MaxFunctionEvaluations', 'MaxFunEvals'}, 100 * n);
if ischar(maxfev) || isstring(maxfev), maxfev = 100 * n; end
tolopt = optim_getopt(options, {'OptimalityTolerance', 'TolFun'}, 1e-6);
tolx = optim_getopt(options, {'StepTolerance', 'TolX'}, 1e-6);
tolf = optim_getopt(options, {'FunctionTolerance'}, 1e-6);
usergrad = fmu_onoff(optim_getopt(options, {'SpecifyObjectiveGradient', 'GradObj'}, false));
fdtype = char(optim_getopt(options, {'FiniteDifferenceType', 'FinDiffType'}, 'forward'));
if strcmpi(fdtype, 'central'), fdstep = eps^(1/3); else, fdstep = sqrt(eps); end
fdstep = optim_getopt(options, {'FiniteDifferenceStepSize'}, fdstep);
lvl = optim_display(options, 'final');

S.fun = fun; S.shape = shape; S.user = usergrad; S.fdtype = fdtype; S.fdstep = fdstep; S.nev = 0;
[f, g, S] = fmu_fg(S, x);
g0 = max(1, norm(g, inf));   % first-order tolerance relative to the initial gradient
Hinv = eye(n);
if lvl == 2
    fprintf('\n Iter  F-count          f(x)    Step-size   First-order optimality\n');
    fprintf(' %4d  %7d  %12.6g                %12.3g\n', 0, S.nev, f, norm(g, inf));
end
exitflag = 0; msg = ''; k = 0; alpha = 0; s = zeros(n, 1);
if norm(g, inf) <= tolopt * g0
    exitflag = 1; msg = 'Local minimum found: the initial point is a stationary point.';
end
OF = optim_outfcn('setup', options);
userstop = false;
if OF.active
    ov = fmu_ov(0, S.nev, f, g, NaN, NaN);
    [s1, OF] = optim_outfcn(OF, reshape(x, shape), ov, 'init');
    [s2, OF] = optim_outfcn(OF, reshape(x, shape), ov, 'iter');
    userstop = s1 || s2;
    if userstop, exitflag = -1; msg = 'Optimization stopped by an output function or plot function.'; end
end
while exitflag == 0
    if k >= maxit
        msg = sprintf('Stopped: the number of iterations reached MaxIterations = %d.', maxit); break
    end
    if S.nev >= maxfev
        msg = sprintf('Stopped: the number of function evaluations reached MaxFunctionEvaluations = %d.', maxfev); break
    end
    k = k + 1;
    p = -Hinv * g;
    if g.' * p >= 0            % not a descent direction (should not happen): restart
        Hinv = eye(n); p = -g;
    end
    a1 = 1;
    if k == 1, a1 = min(1, 1 / norm(g, inf)); end
    [alpha, fnew, gnew, S, lsok] = fmu_wolfe(S, x, f, g, p, a1);
    if ~lsok
        if any(Hinv(:) ~= reshape(eye(n), [], 1))
            Hinv = eye(n); k = k - 1; continue
        end
        exitflag = -2; msg = 'Stopped: the line search could not find a point satisfying the Wolfe conditions.'; break
    end
    s = alpha * p; xnew = x + s; y = gnew - g;
    sy = s.' * y;
    if sy > 1e-12 * norm(s) * norm(y)
        if k == 1, Hinv = (sy / (y.' * y)) * eye(n); end
        rho = 1 / sy;
        Hy = Hinv * y;
        Hinv = Hinv - rho * (s * Hy.' + Hy * s.') + (rho^2 * (y.' * Hy) + rho) * (s * s.');
        Hinv = (Hinv + Hinv.') / 2;
    end
    df = f - fnew;
    x = xnew; f = fnew; g = gnew;
    if lvl == 2
        fprintf(' %4d  %7d  %12.6g  %11.4g  %12.3g\n', k, S.nev, f, alpha, norm(g, inf));
    end
    if OF.active
        [userstop, OF] = optim_outfcn(OF, reshape(x, shape), fmu_ov(k, S.nev, f, g, norm(s), alpha), 'iter');
        if userstop, exitflag = -1; msg = 'Optimization stopped by an output function or plot function.'; break, end
    end
    if f < -1e20
        exitflag = -3; msg = 'Objective function appears unbounded below.'; break
    end
    if norm(g, inf) <= tolopt * g0
        exitflag = 1; msg = 'Local minimum found: the first-order optimality measure is below OptimalityTolerance.'; break
    end
    if norm(s) <= tolx * (1 + norm(x)) && norm(g, inf) <= sqrt(tolopt) * g0
        exitflag = 2; msg = 'Local minimum possible: the step is smaller than StepTolerance.'; break
    end
    if abs(df) <= tolf * eps * (1 + abs(f)) && norm(g, inf) <= 10 * tolopt * g0
        exitflag = 3; msg = 'Local minimum possible: the change in the objective is smaller than FunctionTolerance.'; break
    end
end
x = reshape(x, shape); fval = f; grad = reshape(g, [], 1);
if OF.active, optim_outfcn(OF, x, fmu_ov(k, S.nev, f, g, norm(s), alpha), 'done'); end
hessian = inv(Hinv); hessian = (hessian + hessian.') / 2;
output = struct('iterations', k, 'funcCount', S.nev, 'stepsize', norm(s), 'lssteplength', alpha, ...
    'firstorderopt', norm(g, inf), 'algorithm', 'quasi-newton', 'message', msg);
if lvl == 1 || lvl == 2 || (lvl == -1 && exitflag <= 0)
    fprintf('\n%s\n\n', msg);
end
end

function [f, g, S] = fmu_fg(S, x)
X = reshape(x, S.shape);
if S.user
    [f, g] = S.fun(X); g = double(g(:)); S.nev = S.nev + 1;
else
    f = S.fun(X); S.nev = S.nev + 1;
    [g, nev] = optim_fdjac(S.fun, x, f, S.shape, S.fdtype, S.fdstep);
    g = g(:); S.nev = S.nev + nev;
end
if ~isscalar(f) || ~isreal(f)
    error('MatLivre:fminunc:objNotScalar', 'The objective function must return a real scalar value.');
end
f = double(f);
end

function [a, fa, ga, S, ok] = fmu_wolfe(S, x, f0, g0, p, a1)
% Line search for the strong Wolfe conditions (Nocedal & Wright, Alg. 3.5).
c1 = 1e-4; c2 = 0.9; amax = 1e10;
d0 = g0.' * p;
aprev = 0; fprev = f0; dprev = d0;
a = a1; ok = false; fa = f0; ga = g0;
for i = 1:30
    [fa, ga, S] = fmu_fg(S, x + a * p);
    da = ga.' * p;
    if ~isfinite(fa) || fa > f0 + c1 * a * d0 || (i > 1 && fa >= fprev)
        [a, fa, ga, S, ok] = fmu_zoom(S, x, f0, d0, p, aprev, fprev, dprev, a, fa, da, c1, c2);
        return
    end
    if abs(da) <= -c2 * d0
        ok = true; return
    end
    if da >= 0
        [a, fa, ga, S, ok] = fmu_zoom(S, x, f0, d0, p, a, fa, da, aprev, fprev, dprev, c1, c2);
        return
    end
    aprev = a; fprev = fa; dprev = da;
    a = min(2 * a, amax);
end
end

function [a, fa, ga, S, ok] = fmu_zoom(S, x, f0, d0, p, alo, flo, dlo, ahi, fhi, dhi, c1, c2)
% Nocedal & Wright, Alg. 3.6, with safeguarded cubic interpolation.
ok = false; a = alo; fa = flo; ga = [];
for i = 1:30
    a = fmu_cubic(alo, flo, dlo, ahi, fhi, dhi);
    lo = min(alo, ahi); hi = max(alo, ahi); w = hi - lo;
    if ~isfinite(a) || a < lo + 0.1 * w || a > hi - 0.1 * w, a = (alo + ahi) / 2; end
    [fa, ga, S] = fmu_fg(S, x + a * p);
    da = ga.' * p;
    if ~isfinite(fa) || fa > f0 + c1 * a * d0 || fa >= flo
        ahi = a; fhi = fa; dhi = da;
    else
        if abs(da) <= -c2 * d0
            ok = true; return
        end
        if da * (ahi - alo) >= 0
            ahi = alo; fhi = flo; dhi = dlo;
        end
        alo = a; flo = fa; dlo = da;
    end
    if abs(ahi - alo) < 1e-14 * max(1, abs(alo)), break, end
end
% accept a point with sufficient decrease even if the curvature condition failed
if flo < f0 + c1 * alo * d0 && alo > 0
    a = alo; [fa, ga, S] = fmu_fg(S, x + a * p); ok = true;
end
end

function a = fmu_cubic(a0, f0, d0, a1, f1, d1)
% minimizer of the cubic interpolating f and f' at a0 and a1
if ~all(isfinite([f0 f1 d0 d1])), a = NaN; return, end
e1 = d0 + d1 - 3 * (f0 - f1) / (a0 - a1);
r = e1^2 - d0 * d1;
if r < 0, a = NaN; return, end
e2 = sign(a1 - a0) * sqrt(r);
a = a1 - (a1 - a0) * (d1 + e2 - e1) / (d1 - d0 + 2 * e2);
end

function tf = fmu_onoff(v)
if ischar(v) || isstring(v), tf = strcmpi(v, 'on') || strcmpi(v, 'true'); else, tf = logical(v); end
end

function ov = fmu_ov(k, fc, f, g, stepn, alpha)
ov = struct('iteration', k, 'funccount', fc, 'fval', f, 'gradient', g, 'firstorderopt', norm(g, inf), ...
    'stepsize', stepn, 'lssteplength', alpha, 'procedure', '');
end
