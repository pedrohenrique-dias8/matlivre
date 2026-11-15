function [x, fval, exitflag, output, jacobian] = fsolve(fun, x0, options)
%FSOLVE  Solve a system of nonlinear equations F(x) = 0.
%   x = fsolve(fun, x0)
%   x = fsolve(fun, x0, options)
%   x = fsolve(problem)
%   [x, fval, exitflag, output, jacobian] = fsolve(...)
%
%   Trust-region dogleg method on 0.5*||F(x)||^2: the step combines the Gauss-Newton step
%   (least-squares solution of J*p = -F, so singular or non-square Jacobians are allowed) and the
%   Cauchy step of the model, inside a trust region updated from the ratio of actual to predicted
%   reduction.
%
%   Options: Display, MaxIterations (400), MaxFunctionEvaluations (100*numel(x0)),
%   FunctionTolerance (1e-6), StepTolerance (1e-6), OptimalityTolerance (1e-6),
%   SpecifyObjectiveGradient/Jacobian ('on': fun returns [F, J]), FiniteDifferenceType,
%   FiniteDifferenceStepSize, InitTrustRegionRadius (1).
%   exitflag:  1 equation solved, 2 equation solved (step small), 3 equation solved (change in
%              residual small), 0 limits reached, -2 converged to a point that is not a root,
%             -3 trust region became too small.
%
%   Reference: Nocedal and Wright, Numerical Optimization, Alg. 4.1 and Sec. 4.1 (dogleg) and
%   Ch. 11. Part of the MatLivre optimization toolbox draft (clean-room).
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
tolf = optim_getopt(options, {'FunctionTolerance', 'TolFun'}, 1e-6);
tolx = optim_getopt(options, {'StepTolerance', 'TolX'}, 1e-6);
tolopt = optim_getopt(options, {'OptimalityTolerance'}, 1e-6);
userJ = fsv_onoff(optim_getopt(options, {'SpecifyObjectiveGradient', 'Jacobian'}, false));
fdtype = char(optim_getopt(options, {'FiniteDifferenceType', 'FinDiffType'}, 'forward'));
if strcmpi(fdtype, 'central'), fdstep = eps^(1/3); else, fdstep = sqrt(eps); end
fdstep = optim_getopt(options, {'FiniteDifferenceStepSize'}, fdstep);
delta = optim_getopt(options, {'InitTrustRegionRadius'}, 1) * max(1, norm(x));
lvl = optim_display(options, 'final');

S = struct('fun', fun, 'shape', shape, 'user', userJ, 'fdtype', fdtype, 'fdstep', fdstep, 'nev', 0);
[F, J, S] = fsv_fj(S, x);
fshape = size(F); F = F(:);
m = numel(F);
phi = 0.5 * (F.' * F);
if lvl == 2
    fprintf('\n Iter  F-count    f(x) = ||F||^2   First-order   Trust-region radius\n');
    fprintf(' %4d  %7d  %16.6g  %12.3g  %12.3g\n', 0, S.nev, 2 * phi, norm(J.' * F, inf), delta);
end
exitflag = 0; msg = ''; k = 0; p = zeros(n, 1);
OF = optim_outfcn('setup', options);
userstop = false;
if OF.active
    ov = fsv_ov(0, S.nev, F, J, NaN, delta, fshape);
    [s1, OF] = optim_outfcn(OF, reshape(x, shape), ov, 'init');
    [s2, OF] = optim_outfcn(OF, reshape(x, shape), ov, 'iter');
    userstop = s1 || s2;
end
while ~userstop
    g = J.' * F;
    if norm(F, inf) == 0 || norm(g, inf) <= tolopt * max(1, norm(F, inf))
        if 2 * phi <= tolf
            exitflag = 1; msg = 'Equation solved: the first-order optimality is small and the function values are near zero.';
        else
            exitflag = -2; msg = 'No solution found: converged to a minimum of ||F||^2 that is not a root (the Jacobian may be singular there).';
        end
        break
    end
    if k >= maxit
        msg = sprintf('Stopped: the number of iterations reached MaxIterations = %d.', maxit); break
    end
    if S.nev >= maxfev
        msg = sprintf('Stopped: the number of function evaluations reached MaxFunctionEvaluations = %d.', maxfev); break
    end
    k = k + 1;
    % dogleg step
    if m >= n && fsv_rank_ok(J)
        pgn = -(J \ F);
    else
        pgn = -lsqminnorm(J, F);
    end
    Jg = J * g;
    pc = -((g.' * g) / max(Jg.' * Jg, realmin)) * g;      % minimizer along steepest descent
    if norm(pgn) <= delta
        p = pgn;
    elseif norm(pc) >= delta
        p = -(delta / norm(g)) * g;
    else
        dvec = pgn - pc;                                  % ||pc + t*dvec|| = delta, t in [0, 1]
        a = dvec.' * dvec; b = 2 * pc.' * dvec; c = pc.' * pc - delta^2;
        t = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a);
        p = pc + t * dvec;
    end
    xt = x + p;
    Ft = fsv_callf(S, xt); S.nev = S.nev + 1;
    phit = 0.5 * (Ft.' * Ft);
    pred = phi - 0.5 * sum((F + J * p).^2);
    rho = (phi - phit) / max(pred, realmin);
    if ~isfinite(phit), rho = -1; end
    if rho < 0.25
        delta = 0.25 * norm(p);
    elseif rho > 0.75 && abs(norm(p) - delta) <= 1e-10 * delta
        delta = min(2 * delta, 1e10);
    end
    if rho > 1e-4
        dphi = phi - phit;
        x = xt;
        [F, J, S] = fsv_fj(S, x, Ft);
        F = F(:); phi = phit;
        if lvl == 2
            fprintf(' %4d  %7d  %16.6g  %12.3g  %12.3g\n', k, S.nev, 2 * phi, norm(J.' * F, inf), delta);
        end
        if OF.active
            [userstop, OF] = optim_outfcn(OF, reshape(x, shape), fsv_ov(k, S.nev, F, J, norm(p), delta, fshape), 'iter');
            if userstop, break, end
        end
        if norm(p) <= tolx * (sqrt(eps) + norm(x))
            if 2 * phi <= tolf, exitflag = 2; msg = 'Equation solved: the step is smaller than StepTolerance and the function values are near zero.';
            else, exitflag = -2; msg = 'No solution found: the step is smaller than StepTolerance but the function values are not near zero.'; end
            break
        end
        if dphi <= tolf * eps * max(1, phi) && 2 * phi <= tolf
            exitflag = 3; msg = 'Equation solved: the change in the residual is below FunctionTolerance.'; break
        end
    elseif delta <= tolx * (sqrt(eps) + norm(x))
        if 2 * phi <= tolf, exitflag = 2; msg = 'Equation solved: the trust region is smaller than StepTolerance.';
        else, exitflag = -3; msg = 'No solution found: the trust region radius became too small.'; end
        break
    end
end
if userstop, exitflag = -1; msg = 'Optimization stopped by an output function or plot function.'; end
x = reshape(x, shape); fval = reshape(F, fshape); jacobian = J;
if OF.active, optim_outfcn(OF, x, fsv_ov(k, S.nev, F, J, norm(p), delta, fshape), 'done'); end
output = struct('iterations', k, 'funcCount', S.nev, 'algorithm', 'trust-region-dogleg', ...
    'firstorderopt', norm(J.' * F, inf), 'stepsize', norm(p), 'message', msg);
if lvl == 1 || lvl == 2 || (lvl == -1 && exitflag <= 0)
    fprintf('\n%s\n\n', msg);
end
end

function tf = fsv_rank_ok(J)
tf = rcond(J.' * J) > 1e-14;
end

function F = fsv_callf(S, x)
F = S.fun(reshape(x, S.shape)); F = double(F(:));
end

function [F, J, S] = fsv_fj(S, x, F)
X = reshape(x, S.shape);
if S.user
    [F, J] = S.fun(X); F = double(F); J = double(J); S.nev = S.nev + 1;
else
    if nargin < 3
        F = S.fun(X); F = double(F); S.nev = S.nev + 1;
    end
    [J, nev] = optim_fdjac(S.fun, x, F(:), S.shape, S.fdtype, S.fdstep);
    S.nev = S.nev + nev;
end
end

function tf = fsv_onoff(v)
if ischar(v) || isstring(v), tf = strcmpi(v, 'on') || strcmpi(v, 'true'); else, tf = logical(v); end
end

function ov = fsv_ov(k, fc, F, J, stepn, delta, fshape)
ov = struct('iteration', k, 'funccount', fc, 'fval', reshape(F, fshape), 'residual', F, ...
    'gradient', J.' * F, 'firstorderopt', norm(J.' * F, inf), 'stepsize', stepn, ...
    'trustregionradius', delta, 'procedure', '');
end
