function [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(fun, x0, lb, ub, options)
%LSQNONLIN  Nonlinear least squares with bounds.
%   x = lsqnonlin(fun, x0)                minimizes sum(fun(x).^2)
%   x = lsqnonlin(fun, x0, lb, ub)        subject to lb <= x <= ub
%   x = lsqnonlin(fun, x0, lb, ub, options)
%   x = lsqnonlin(problem)
%   [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(...)
%
%   Levenberg-Marquardt method with Marquardt's diagonal scaling, Nielsen's damping update and
%   bounds handled by projection (the step is projected onto the box and the gain ratio is
%   computed for the projected step). Every iterate satisfies the bounds.
%
%   Options: Display, MaxIterations (400), MaxFunctionEvaluations (100*numel(x0)),
%   FunctionTolerance (1e-6), StepTolerance (1e-6), OptimalityTolerance (1e-6),
%   SpecifyObjectiveGradient/Jacobian ('on': fun returns [F, J]), FiniteDifferenceType,
%   FiniteDifferenceStepSize, InitDamping (0.01).
%   exitflag: 1 first-order optimality small, 2 step small, 3 change in resnorm small,
%             0 limits reached, -2 lb > ub.
%   lambda.lower / lambda.upper are the multipliers of the active bounds.
%
%   Reference: Nocedal and Wright, Numerical Optimization, Sec. 10.3; J. J. More (1978), "The
%   Levenberg-Marquardt algorithm: implementation and theory"; H. B. Nielsen (1999), "Damping
%   parameter in Marquardt's method". Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(fun)
    p = fun; fun = p.objective; x0 = p.x0;
    lb = lsn_fld(p, 'lb'); ub = lsn_fld(p, 'ub'); options = lsn_fld(p, 'options');
    nin = 5;
else
    nin = nargin;
end
if nin < 3, lb = []; end
if nin < 4, ub = []; end
if nin < 5 || isempty(options), options = struct(); end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
shape = size(x0); x = double(x0(:)); n = numel(x);
if isempty(lb), lb = -inf(n, 1); end
if isempty(ub), ub = inf(n, 1); end
lb = double(lb(:)); ub = double(ub(:));
maxit = optim_getopt(options, {'MaxIterations', 'MaxIter'}, 400);
maxfev = optim_getopt(options, {'MaxFunctionEvaluations', 'MaxFunEvals'}, 100 * n);
if ischar(maxfev) || isstring(maxfev), maxfev = 100 * n; end
tolf = optim_getopt(options, {'FunctionTolerance', 'TolFun'}, 1e-6);
tolx = optim_getopt(options, {'StepTolerance', 'TolX'}, 1e-6);
tolopt = optim_getopt(options, {'OptimalityTolerance'}, 1e-6);
userJ = lsn_onoff(optim_getopt(options, {'SpecifyObjectiveGradient', 'Jacobian'}, false));
fdtype = char(optim_getopt(options, {'FiniteDifferenceType', 'FinDiffType'}, 'forward'));
if strcmpi(fdtype, 'central'), fdstep = eps^(1/3); else, fdstep = sqrt(eps); end
fdstep = optim_getopt(options, {'FiniteDifferenceStepSize'}, fdstep);
mu = optim_getopt(options, {'InitDamping'}, 1e-2);
lvl = optim_display(options, 'final');

output = struct('iterations', 0, 'funcCount', 0, 'firstorderopt', NaN, 'stepsize', NaN, ...
    'algorithm', 'levenberg-marquardt', 'message', '');
if any(lb > ub)
    exitflag = -2; resnorm = []; residual = []; jacobian = []; lambda = [];
    output.message = 'The bounds are inconsistent: lb(i) > ub(i) for some i.';
    x = reshape(x, shape);
    if lvl ~= 0, fprintf('\n%s\n\n', output.message); end
    return
end
x = min(max(x, lb), ub);
S = struct('fun', fun, 'shape', shape, 'user', userJ, 'fdtype', fdtype, 'fdstep', fdstep, ...
    'lb', lb, 'ub', ub, 'nev', 0);
[r, J, S] = lsn_fj(S, x, true);
f = r.' * r;
g = J.' * r;
DtD = max(diag(J.' * J), 1e-12);
damp = mu * max(DtD);
nu = 2;
if lvl == 2
    fprintf('\n Iter  F-count     Resnorm   First-order    Lambda\n');
    fprintf(' %4d  %7d  %10.6g  %12.3g  %8.3g\n', 0, S.nev, f, norm(lsn_pgrad(x, g, lb, ub), inf), damp);
end
exitflag = 0; msg = ''; k = 0; s = zeros(n, 1);
OF = optim_outfcn('setup', options);
userstop = false;
if OF.active
    ov = lsn_ov(0, S.nev, f, r, g, x, lb, ub, NaN, damp);
    [s1, OF] = optim_outfcn(OF, reshape(x, shape), ov, 'init');
    [s2, OF] = optim_outfcn(OF, reshape(x, shape), ov, 'iter');
    userstop = s1 || s2;
end
while ~userstop
    fo = norm(2 * lsn_pgrad(x, g, lb, ub), inf);
    if fo <= tolopt * max(1, f)
        exitflag = 1; msg = 'Local minimum found: the first-order optimality measure is below OptimalityTolerance.'; break
    end
    if k >= maxit
        msg = sprintf('Stopped: the number of iterations reached MaxIterations = %d.', maxit); break
    end
    if S.nev >= maxfev
        msg = sprintf('Stopped: the number of function evaluations reached MaxFunctionEvaluations = %d.', maxfev); break
    end
    k = k + 1;
    accepted = false;
    for inner = 1:60
        % damped Gauss-Newton step as a least-squares problem (avoids forming J'*J)
        % variables held at a bound by the gradient are fixed for this step (active set)
        free = ~((x <= lb & g > 0) | (x >= ub & g < 0));
        nf = sum(free);
        h = zeros(n, 1);
        h(free) = -[J(:, free); diag(sqrt(damp * DtD(free)))] \ [r; zeros(nf, 1)];
        xt = min(max(x + h, lb), ub);
        s = xt - x;
        if norm(s) <= tolx * (sqrt(eps) + norm(x)) * 1e-3
            break
        end
        rt = lsn_callf(S, xt); S.nev = S.nev + 1;
        ft = rt.' * rt;
        pred = f - sum((r + J * s).^2);
        rho = (f - ft) / max(pred, realmin);
        if isfinite(ft) && ft < f && rho > 1e-4
            accepted = true;
            damp = damp * max(1 / 3, 1 - (2 * rho - 1)^3); nu = 2;
            break
        end
        damp = damp * nu; nu = 2 * nu;
        if S.nev >= maxfev, break, end
    end
    if ~accepted
        if S.nev >= maxfev
            msg = sprintf('Stopped: the number of function evaluations reached MaxFunctionEvaluations = %d.', maxfev); break
        end
        exitflag = 2; msg = 'Local minimum possible: no step that reduces the sum of squares is larger than StepTolerance.'; break
    end
    df = f - ft;
    x = xt;
    [r, J, S] = lsn_fj(S, x, true, rt);
    f = ft; g = J.' * r;
    DtD = max(DtD, diag(J.' * J));
    if lvl == 2
        fprintf(' %4d  %7d  %10.6g  %12.3g  %8.3g\n', k, S.nev, f, norm(2 * lsn_pgrad(x, g, lb, ub), inf), damp);
    end
    if OF.active
        [userstop, OF] = optim_outfcn(OF, reshape(x, shape), lsn_ov(k, S.nev, f, r, g, x, lb, ub, norm(s), damp), 'iter');
        if userstop, break, end
    end
    if norm(s) <= tolx * (sqrt(eps) + norm(x))
        exitflag = 2; msg = 'Local minimum possible: the step is smaller than StepTolerance.'; break
    end
    if df <= tolf * f
        exitflag = 3; msg = 'Local minimum possible: the relative change in the sum of squares is below FunctionTolerance.'; break
    end
end
if userstop, exitflag = -1; msg = 'Optimization stopped by an output function or plot function.'; end
resnorm = f; residual = r; jacobian = J;
if OF.active, optim_outfcn(OF, reshape(x, shape), lsn_ov(k, S.nev, f, r, g, x, lb, ub, norm(s), damp), 'done'); end
fo = norm(2 * lsn_pgrad(x, g, lb, ub), inf);
lambda.lower = zeros(n, 1); lambda.upper = zeros(n, 1);
atl = x <= lb & 2 * g > 0; atu = x >= ub & 2 * g < 0;
lambda.lower(atl) = 2 * g(atl); lambda.upper(atu) = -2 * g(atu);
output.iterations = k; output.funcCount = S.nev; output.firstorderopt = fo;
output.stepsize = norm(s); output.message = msg;
x = reshape(x, shape);
if lvl == 1 || lvl == 2 || (lvl == -1 && exitflag <= 0)
    fprintf('\n%s\n\n', msg);
end
end

function pg = lsn_pgrad(x, g, lb, ub)
% projected gradient: zero for components pushing against an active bound
pg = x - min(max(x - g, lb), ub);
end

function r = lsn_callf(S, x)
r = S.fun(reshape(x, S.shape)); r = double(r(:));
end

function [r, J, S] = lsn_fj(S, x, ~, r)
X = reshape(x, S.shape);
if S.user
    [r, J] = S.fun(X); r = double(r(:)); J = double(J); S.nev = S.nev + 1;
else
    if nargin < 4
        r = lsn_callf(S, x); S.nev = S.nev + 1;
    end
    [J, nev] = optim_fdjac(S.fun, x, r, S.shape, S.fdtype, S.fdstep, S.lb, S.ub);
    S.nev = S.nev + nev;
end
end

function tf = lsn_onoff(v)
if ischar(v) || isstring(v), tf = strcmpi(v, 'on') || strcmpi(v, 'true'); else, tf = logical(v); end
end

function v = lsn_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end

function ov = lsn_ov(k, fc, f, r, g, x, lb, ub, stepn, damp)
ov = struct('iteration', k, 'funccount', fc, 'fval', f, 'resnorm', f, 'residual', r, ...
    'gradient', 2 * g, 'firstorderopt', norm(2 * lsn_pgrad(x, g, lb, ub), inf), 'stepsize', stepn, ...
    'lambda', damp, 'procedure', '');
end
