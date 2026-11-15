function [x, fval, exitflag, output, lambda, grad, hessian] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
%FMINCON  Find a minimum of a constrained nonlinear multivariable function.
%
%   x = fmincon(fun, x0, A, b)
%   x = fmincon(fun, x0, A, b, Aeq, beq)
%   x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub)
%   x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon)
%   x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
%   x = fmincon(problem)
%   [x, fval, exitflag, output, lambda, grad, hessian] = fmincon(...)
%
%   minimizes fun(x) subject to A*x <= b, Aeq*x = beq, lb <= x <= ub, c(x) <= 0, ceq(x) = 0,
%   where [c, ceq] = nonlcon(x). Pass [] for absent arguments.
%
%   Algorithm: sequential quadratic programming (SQP) with a damped BFGS approximation of the
%   Hessian of the Lagrangian, an elastic QP subproblem solved by a primal-dual interior-point
%   method (optim_qpip), an l1 exact-penalty merit function with backtracking line search and a
%   second-order correction step. Iterates always satisfy the bounds.
%
%   Options (struct from optimoptions or optimset; both name styles are accepted):
%     Display                    'off' | 'iter' | 'final' (default) | 'notify'
%     MaxIterations / MaxIter    default 400
%     MaxFunctionEvaluations / MaxFunEvals   default 100*numel(x0)
%     OptimalityTolerance / TolFun           default 1e-6
%     ConstraintTolerance / TolCon           default 1e-6
%     StepTolerance / TolX                   default 1e-10
%     SpecifyObjectiveGradient / GradObj     fun returns [f, g]           (default false)
%     SpecifyConstraintGradient / GradConstr nonlcon returns [c, ceq, GC, GCeq], gradients in
%                                            columns (n-by-m)               (default false)
%     FiniteDifferenceType       'forward' (default) | 'central'
%     FiniteDifferenceStepSize   default sqrt(eps) forward, eps^(1/3) central
%     Algorithm                  only 'sqp' is implemented; other names run 'sqp'
%
%   exitflag:  1 first-order optimality and feasibility reached
%              2 step smaller than StepTolerance with the constraints satisfied
%              0 MaxIterations or MaxFunctionEvaluations reached
%             -2 no feasible point found
%             -3 line search failed (no further progress possible)
%
%   Reference: J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed., Ch. 18
%   (SQP, Powell-damped BFGS, l1 merit, second-order correction) and Sec. 16.6 (QP).
%   Clean-room implementation for the MatLivre optimization toolbox draft.

% ------------------------------------------------------------------ arguments
if nargin == 1 && isstruct(fun)
    pr = fun;
    fun = fmc_pfield(pr, 'objective', []); x0 = fmc_pfield(pr, 'x0', []);
    A = fmc_pfield(pr, 'Aineq', []); b = fmc_pfield(pr, 'bineq', []);
    Aeq = fmc_pfield(pr, 'Aeq', []); beq = fmc_pfield(pr, 'beq', []);
    lb = fmc_pfield(pr, 'lb', []); ub = fmc_pfield(pr, 'ub', []);
    nonlcon = fmc_pfield(pr, 'nonlcon', []); options = fmc_pfield(pr, 'options', struct());
    nargs = 10;
else
    nargs = nargin;
end
if nargs < 2, error('MatLivre:fmincon:notEnoughInputs', 'fmincon requires at least two input arguments.'); end
if nargs < 3, A = []; end
if nargs < 4, b = []; end
if nargs < 5, Aeq = []; end
if nargs < 6, beq = []; end
if nargs < 7, lb = []; end
if nargs < 8, ub = []; end
if nargs < 9, nonlcon = []; end
if nargs < 10 || isempty(options), options = struct(); end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
if ischar(nonlcon) || isstring(nonlcon), nonlcon = str2func(nonlcon); end
if isempty(x0), error('MatLivre:fmincon:emptyX0', 'x0 must not be empty.'); end

xshape = size(x0);
x = double(x0(:));
n = numel(x);
if isempty(A), A = zeros(0, n); b = zeros(0, 1); end
if isempty(Aeq), Aeq = zeros(0, n); beq = zeros(0, 1); end
b = double(b(:)); beq = double(beq(:));
if size(A, 2) ~= n || size(A, 1) ~= numel(b)
    error('MatLivre:fmincon:sizeA', 'A must have numel(x0) columns and as many rows as b has elements.');
end
if size(Aeq, 2) ~= n || size(Aeq, 1) ~= numel(beq)
    error('MatLivre:fmincon:sizeAeq', 'Aeq must have numel(x0) columns and as many rows as beq has elements.');
end
if isempty(lb), lb = -inf(n, 1); end
if isempty(ub), ub = inf(n, 1); end
lb = double(lb(:)); ub = double(ub(:));
if numel(lb) < n, lb = [lb; -inf(n - numel(lb), 1)]; end
if numel(ub) < n, ub = [ub; inf(n - numel(ub), 1)]; end
lb = lb(1:n); ub = ub(1:n);

% ------------------------------------------------------------------ options
o.display = lower(char(fmc_opt(options, {'Display'}, 'final')));
o.maxit = fmc_opt(options, {'MaxIterations', 'MaxIter'}, 400);
o.maxfev = fmc_opt(options, {'MaxFunctionEvaluations', 'MaxFunEvals'}, 100 * n);
if ischar(o.maxfev) || isstring(o.maxfev), o.maxfev = 100 * n; end
o.tolopt = fmc_opt(options, {'OptimalityTolerance', 'TolFun'}, 1e-6);
o.tolcon = fmc_opt(options, {'ConstraintTolerance', 'TolCon'}, 1e-6);
o.tolx = fmc_opt(options, {'StepTolerance', 'TolX'}, 1e-10);
o.gradobj = fmc_onoff(fmc_opt(options, {'SpecifyObjectiveGradient', 'GradObj'}, false));
o.gradcon = fmc_onoff(fmc_opt(options, {'SpecifyConstraintGradient', 'GradConstr'}, false));
o.fdtype = lower(char(fmc_opt(options, {'FiniteDifferenceType', 'FinDiffType'}, 'forward')));
if strcmp(o.fdtype, 'central'), defh = eps^(1/3); else, defh = sqrt(eps); end
o.fdstep = fmc_opt(options, {'FiniteDifferenceStepSize'}, defh);
verbose = 1;
switch o.display
    case {'off', 'none'}, verbose = 0;
    case {'iter', 'iter-detailed'}, verbose = 2;
    case {'notify', 'notify-detailed'}, verbose = -1;
end

output = struct('iterations', 0, 'funcCount', 0, 'constrviolation', NaN, 'stepsize', NaN, ...
    'algorithm', 'sqp', 'firstorderopt', NaN, 'message', '', 'lssteplength', NaN);

if any(lb > ub)
    fval = NaN; exitflag = -2; lambda = []; grad = []; hessian = [];
    x = reshape(x, xshape);
    output.message = 'The bounds are inconsistent: lb(i) > ub(i) for some i.';
    if verbose ~= 0, fprintf('\n%s\n', output.message); end
    return
end

% iterates stay inside the bounds
x = min(max(x, lb), ub);

ilb = find(isfinite(lb)); iub = find(isfinite(ub));
Jb = [-fmc_sparse_rows(ilb, n); fmc_sparse_rows(iub, n)];
mA = size(A, 1); mAe = size(Aeq, 1);

P = struct('fun', fun, 'nonlcon', nonlcon, 'shape', xshape, 'A', A, 'b', b, 'Aeq', Aeq, ...
    'beq', beq, 'lb', lb, 'ub', ub, 'o', o, 'n', n, 'fcount', 0);

% ------------------------------------------------------------------ first evaluation
[ev, P] = fmc_evaluate(P, x, true);
mc = numel(ev.c); mce = numel(ev.ceq);
mg = mA + mc;           % general inequalities (elastic)
me = mAe + mce;         % equalities (elastic)
B = eye(n);
mu = 1;
lam_g = zeros(mg, 1); lam_eq = zeros(me, 1); lam_b = zeros(numel(ilb) + numel(iub), 1);
gscale = 1;   % absolute first-order tolerance (gradient of the Lagrangian, infinity norm)
viol = fmc_violation(ev, x, lb, ub);
if verbose == 2
    fprintf('\n Iter  Func-count            Fval   Feasibility   Step Length     Norm of   First-order\n');
    fprintf('                                                                   step    optimality\n');
    fprintf(' %4d  %10d  %14.6e  %12.3e\n', 0, P.fcount, ev.f, viol);
end

exitflag = 0;
msg = '';
k = 0;
stepnorm = NaN; alpha = NaN; fo = NaN;
breset = false;
OF = optim_outfcn('setup', options);
userstop = false;
if OF.active
    ov = fmc_optimvalues(0, P.fcount, ev, viol, NaN, NaN, NaN);
    [s1, OF] = optim_outfcn(OF, reshape(x, xshape), ov, 'init');
    [s2, OF] = optim_outfcn(OF, reshape(x, xshape), ov, 'iter');
    userstop = s1 || s2;
end
while ~userstop
    if k >= o.maxit
        exitflag = 0; msg = sprintf('Stopped: the number of iterations reached MaxIterations = %d.', o.maxit);
        break
    end
    if P.fcount >= o.maxfev
        exitflag = 0; msg = sprintf('Stopped: the number of function evaluations reached MaxFunctionEvaluations = %d.', o.maxfev);
        break
    end
    k = k + 1;

    % ---- QP subproblem (elastic in the general constraints, exact in the bounds)
    cg = [A * x - b; ev.c];   Jg = [A; ev.Jc];
    ce = [Aeq * x - beq; ev.ceq]; Je = [Aeq; ev.Jceq];
    cb = [lb(ilb) - x(ilb); x(iub) - ub(iub)];
    rho = max([1e3 * gscale, 10 * mu, 1e3]);
    nz = n + mg + 2 * me;
    Hq = zeros(nz); Hq(1:n, 1:n) = B;
    fq = [ev.g; rho * ones(mg + 2 * me, 1)];
    Ai = [Jg, -eye(mg), zeros(mg, 2 * me);
          zeros(mg, n), -eye(mg), zeros(mg, 2 * me);
          Jb, zeros(size(Jb, 1), mg + 2 * me);
          zeros(2 * me, n + mg), -eye(2 * me)];
    bi = [-cg; zeros(mg, 1); -cb; zeros(2 * me, 1)];
    Ae = [Je, zeros(me, mg), -eye(me), eye(me)];
    [z, yq, zq, qinfo] = optim_qpip(Hq, fq, Ai, bi, Ae, -ce, struct('tol', 1e-12, 'maxit', 300));
    d = z(1:n);
    lam_g_new = zq(1:mg);
    lam_b_new = zq(2 * mg + 1:2 * mg + numel(cb));
    lam_eq_new = yq;
    t_el = z(n + 1:n + mg); uv = z(n + mg + 1:end);

    % ---- merit function and penalty parameter
    lmax = max([abs(lam_g_new); abs(lam_eq_new); 0]);
    if mu < 1.1 * lmax, mu = max(1.5 * mu, 1.1 * lmax); end
    v1 = sum(max(0, cg)) + sum(abs(ce));
    v1lin = sum(max(0, cg + Jg * d)) + sum(abs(ce + Je * d));
    phi0 = ev.f + mu * v1;
    D = ev.g.' * d - mu * (v1 - v1lin);
    if D > -1e-14 * (1 + abs(phi0)), D = -1e-14 * (1 + abs(phi0)); end
    stepnorm = norm(d);

    % converged on the step? (tiny QP step at a feasible point)
    if stepnorm <= o.tolx * (1 + norm(x)) && viol <= o.tolcon
        lam_g = lam_g_new; lam_eq = lam_eq_new; lam_b = lam_b_new;
        fo = fmc_firstorder(ev, lam_g, lam_eq, lam_b, Jg, Je, Jb, cg);
        if fo <= o.tolopt * gscale
            exitflag = 1; msg = 'Local minimum found that satisfies the constraints: first-order optimality is below OptimalityTolerance and the constraints are satisfied within ConstraintTolerance.';
        else
            exitflag = 2; msg = 'Local minimum possible: the step is smaller than StepTolerance and the constraints are satisfied.';
        end
        alpha = 0;
        break
    end
    % stationary point of the infeasibility: the elastic QP cannot reduce the violation further
    if stepnorm <= sqrt(o.tolx) * (1 + norm(x)) && viol > o.tolcon && sum(t_el) + sum(uv) > o.tolcon
        exitflag = -2; msg = 'No feasible point found: converged to an infeasible point where the constraint violation cannot be reduced.';
        alpha = 0;
        break
    end

    % ---- line search on the l1 merit function (with one second-order correction)
    alpha = 1; accepted = false; xt = x; evt = ev;
    for ls = 1:40
        xt = min(max(x + alpha * d, lb), ub);
        [evt, P] = fmc_evaluate(P, xt, false);
        phit = evt.f + mu * fmc_merit_viol(evt, xt, P);
        if phit <= phi0 + 1e-4 * alpha * D && isfinite(phit)
            accepted = true;
            break
        end
        if ls == 1 && (me + mg) > 0 && isfinite(phit)
            % second-order correction (Maratos effect)
            cgt = [A * xt - b; evt.c]; cet = [Aeq * xt - beq; evt.ceq];
            act = find(lam_g_new > 1e-10 | cgt > 0);
            J = [Je; Jg(act, :)]; cc = [cet; cgt(act)];
            if ~isempty(cc)
                dsoc = d - lsqminnorm(J, cc);   % minimum-norm correction onto the linearized active set
                xs = min(max(x + dsoc, lb), ub);
                [evs, P] = fmc_evaluate(P, xs, false);
                phis = evs.f + mu * fmc_merit_viol(evs, xs, P);
                if phis <= phi0 + 1e-4 * D && isfinite(phis)
                    xt = xs; evt = evs; accepted = true;
                    break
                end
            end
        end
        % backtrack (safeguarded quadratic interpolation)
        if isfinite(phit)
            anew = -D * alpha^2 / (2 * (phit - phi0 - D * alpha));
            alpha = min(0.5 * alpha, max(0.1 * alpha, anew));
        else
            alpha = 0.1 * alpha;
        end
        if alpha < 1e-12, break, end
    end
    if ~accepted
        if ~breset
            B = eye(n); breset = true; k = k - 1;   % retry the iteration with a fresh Hessian
            continue
        end
        if viol <= o.tolcon
            exitflag = 2; msg = 'Local minimum possible: the line search cannot improve the point further and the constraints are satisfied.';
        else
            exitflag = -3; msg = 'Stopped: the line search failed to reduce the merit function.';
        end
        break
    end
    breset = false;

    % ---- gradients at the new point, BFGS update
    [evt, P] = fmc_evaluate(P, xt, true, evt);
    Jg_new = [A; evt.Jc]; Je_new = [Aeq; evt.Jceq];
    s = xt - x;
    gl_old = ev.g + Jg.' * lam_g_new + Je.' * lam_eq_new;
    gl_new = evt.g + Jg_new.' * lam_g_new + Je_new.' * lam_eq_new;
    yv = gl_new - gl_old;
    sy = s.' * yv;
    if k == 1 && sy > 0
        B = (yv.' * yv) / sy * eye(n);
    end
    Bs = B * s; sBs = s.' * Bs;
    if sBs > 0
        if sy < 0.2 * sBs
            th = 0.8 * sBs / (sBs - sy);
            r = th * yv + (1 - th) * Bs;
        else
            r = yv;
        end
        sr = s.' * r;
        if sr > 1e-16 * sBs
            B = B - (Bs * Bs.') / sBs + (r * r.') / sr;
            B = (B + B.') / 2;
        end
    end

    x = xt; ev = evt;
    lam_g = lam_g_new; lam_eq = lam_eq_new; lam_b = lam_b_new;
    cg = [A * x - b; ev.c]; Jg = Jg_new; Je = Je_new;
    viol = fmc_violation(ev, x, lb, ub);
    fo = fmc_firstorder(ev, lam_g, lam_eq, lam_b, Jg, Je, Jb, cg);
    stepnorm = norm(s);
    if verbose == 2
        fprintf(' %4d  %10d  %14.6e  %12.3e  %12.3e  %10.3e  %12.3e\n', k, P.fcount, ev.f, viol, alpha, stepnorm, fo);
    end
    if OF.active
        [userstop, OF] = optim_outfcn(OF, reshape(x, xshape), fmc_optimvalues(k, P.fcount, ev, viol, fo, stepnorm, alpha), 'iter');
        if userstop, break, end
    end
    if fo <= o.tolopt * gscale && viol <= o.tolcon
        exitflag = 1; msg = 'Local minimum found that satisfies the constraints: first-order optimality is below OptimalityTolerance and the constraints are satisfied within ConstraintTolerance.';
        break
    end
    if stepnorm <= o.tolx * (1 + norm(x)) && viol <= o.tolcon
        exitflag = 2; msg = 'Local minimum possible: the step is smaller than StepTolerance and the constraints are satisfied.';
        break
    end
end

if userstop
    exitflag = -1; msg = 'Optimization stopped by an output function or plot function.';
end
if exitflag == 0 && viol > o.tolcon
    msg = [msg, ' The constraints are not satisfied.'];
end
if exitflag ~= 0 && exitflag ~= -3 && exitflag ~= -1 && viol > o.tolcon
    exitflag = -2; msg = 'No feasible point found: the constraints are violated by more than ConstraintTolerance.';
end
if ~isfinite(fo)
    fo = fmc_firstorder(ev, lam_g, lam_eq, lam_b, [A; ev.Jc], [Aeq; ev.Jceq], Jb, [A * x - b; ev.c]);
end

fval = ev.f;
grad = reshape(ev.g, [], 1);
hessian = B;
output.iterations = k;
output.funcCount = P.fcount;
output.constrviolation = viol;
output.stepsize = stepnorm;
output.lssteplength = alpha;
output.firstorderopt = fo;
output.message = msg;
lambda.lower = zeros(n, 1); lambda.upper = zeros(n, 1);
nlb = numel(ilb);
lambda.lower(ilb) = lam_b(1:nlb);
lambda.upper(iub) = lam_b(nlb + 1:end);
lambda.ineqlin = lam_g(1:mA);
lambda.eqlin = lam_eq(1:mAe);
lambda.ineqnonlin = lam_g(mA + 1:end);
lambda.eqnonlin = lam_eq(mAe + 1:end);
x = reshape(x, xshape);
if OF.active
    optim_outfcn(OF, x, fmc_optimvalues(k, P.fcount, ev, viol, fo, stepnorm, alpha), 'done');
end
if verbose == 1 || (verbose == -1 && exitflag <= 0)
    fprintf('\n%s\n\n', msg);
end
end

% ====================================================================== helpers

function [ev, P] = fmc_evaluate(P, x, needgrad, ev)
% Objective and constraints at x; gradients by the user or by finite differences.
if nargin < 4 || isempty(ev)
    ev = struct();
    [ev.f, gu, P] = fmc_call_fun(P, x, needgrad && P.o.gradobj);
    if needgrad && P.o.gradobj, ev.g = double(gu(:)); end
    [ev.c, ev.ceq, Jc, Jce] = fmc_call_con(P, x, needgrad && P.o.gradcon);
    if needgrad && P.o.gradcon, ev.Jc = Jc; ev.Jceq = Jce; end
end
if ~needgrad
    ev.g = []; ev.Jc = []; ev.Jceq = [];
    return
end
if nargin >= 4 && ~isempty(ev)
    % point already evaluated without gradients: get the user-supplied ones now
    if P.o.gradobj, [~, gu, P] = fmc_call_fun(P, x, true); ev.g = double(gu(:)); end
    if P.o.gradcon, [~, ~, Jc, Jce] = fmc_call_con(P, x, true); ev.Jc = Jc; ev.Jceq = Jce; end
end
needfg = ~P.o.gradobj; needcg = ~P.o.gradcon && ~isempty(P.nonlcon);
if ~needcg && ~P.o.gradcon
    ev.Jc = zeros(numel(ev.c), P.n); ev.Jceq = zeros(numel(ev.ceq), P.n);
end
if needfg || needcg
    n = P.n;
    g = zeros(n, 1); Jc = zeros(numel(ev.c), n); Jce = zeros(numel(ev.ceq), n);
    central = strcmp(P.o.fdtype, 'central');
    hs = P.o.fdstep;
    for j = 1:n
        if numel(hs) > 1, hj = hs(j); else, hj = hs; end
        h = hj * max(abs(x(j)), 1);
        if central
            if x(j) + h > P.ub(j) || x(j) - h < P.lb(j)
                % one-sided near a bound
                if x(j) + h > P.ub(j), h = -h; end
                xp = x; xp(j) = x(j) + h;
                [fp, cp, cep, P] = fmc_fd_eval(P, xp, needfg, needcg);
                if needfg, g(j) = (fp - ev.f) / h; end
                if needcg, Jc(:, j) = (cp - ev.c) / h; Jce(:, j) = (cep - ev.ceq) / h; end
            else
                xp = x; xp(j) = x(j) + h; xm = x; xm(j) = x(j) - h;
                [fp, cp, cep, P] = fmc_fd_eval(P, xp, needfg, needcg);
                [fm, cm, cem, P] = fmc_fd_eval(P, xm, needfg, needcg);
                if needfg, g(j) = (fp - fm) / (2 * h); end
                if needcg, Jc(:, j) = (cp - cm) / (2 * h); Jce(:, j) = (cep - cem) / (2 * h); end
            end
        else
            if x(j) + h > P.ub(j), h = -h; end
            xp = x; xp(j) = x(j) + h;
            [fp, cp, cep, P] = fmc_fd_eval(P, xp, needfg, needcg);
            if needfg, g(j) = (fp - ev.f) / h; end
            if needcg, Jc(:, j) = (cp - ev.c) / h; Jce(:, j) = (cep - ev.ceq) / h; end
        end
    end
    if needfg, ev.g = g; end
    if needcg, ev.Jc = Jc; ev.Jceq = Jce; end
end
end

function [fp, cp, cep, P] = fmc_fd_eval(P, x, needf, needc)
fp = []; cp = []; cep = [];
if needf, [fp, ~, P] = fmc_call_fun(P, x, false); end
if needc, [cp, cep] = fmc_call_con(P, x, false); end
end

function [f, g, P] = fmc_call_fun(P, x, wantg)
X = reshape(x, P.shape);
g = [];
if wantg
    [f, g] = P.fun(X);
else
    f = P.fun(X);
end
P.fcount = P.fcount + 1;
if ~isscalar(f) || ~isreal(f)
    error('MatLivre:fmincon:objNotScalar', 'The objective function must return a real scalar value.');
end
f = double(f);
end

function [c, ceq, Jc, Jce] = fmc_call_con(P, x, wantg)
Jc = []; Jce = [];
if isempty(P.nonlcon)
    c = zeros(0, 1); ceq = zeros(0, 1);
    Jc = zeros(0, P.n); Jce = zeros(0, P.n);
    return
end
X = reshape(x, P.shape);
if wantg
    [c, ceq, GC, GCe] = P.nonlcon(X);
    if isempty(GC), Jc = zeros(numel(c), P.n); else, Jc = double(GC).'; end
    if isempty(GCe), Jce = zeros(numel(ceq), P.n); else, Jce = double(GCe).'; end
else
    [c, ceq] = P.nonlcon(X);
end
c = double(c(:)); ceq = double(ceq(:));
end

function v = fmc_merit_viol(ev, x, P)
v = sum(max(0, P.A * x - P.b)) + sum(max(0, ev.c)) + sum(abs(P.Aeq * x - P.beq)) + sum(abs(ev.ceq));
end

function v = fmc_violation(ev, x, lb, ub)
v = max([0; ev.c; abs(ev.ceq); lb - x; x - ub]);
end

function fo = fmc_firstorder(ev, lam_g, lam_eq, lam_b, Jg, Je, Jb, cg)
% Infinity norm of the gradient of the Lagrangian and of the complementarity products.
gl = ev.g + Jg.' * lam_g + Je.' * lam_eq;
if ~isempty(lam_b), gl = gl + Jb.' * lam_b; end
compl = 0;
if ~isempty(lam_g), compl = max(abs(lam_g .* min(cg, 0))); end
fo = max(norm(gl, inf), compl);
end

function M = fmc_sparse_rows(idx, n)
M = zeros(numel(idx), n);
for i = 1:numel(idx), M(i, idx(i)) = 1; end
end

function v = fmc_opt(s, names, default)
v = default;
if ~isstruct(s), return, end
for i = 1:numel(names)
    if isfield(s, names{i}) && ~isempty(s.(names{i}))
        v = s.(names{i});
        return
    end
end
end

function tf = fmc_onoff(v)
if ischar(v) || isstring(v)
    tf = strcmpi(v, 'on') || strcmpi(v, 'true');
else
    tf = logical(v);
end
end

function v = fmc_pfield(s, name, default)
if isfield(s, name), v = s.(name); else, v = default; end
end

function ov = fmc_optimvalues(k, fc, ev, viol, fo, stepnorm, alpha)
% optimValues passed to OutputFcn / PlotFcn
ov = struct('iteration', k, 'funccount', fc, 'fval', ev.f, 'constrviolation', viol, ...
    'firstorderopt', fo, 'stepsize', stepnorm, 'lssteplength', alpha, 'gradient', ev.g, ...
    'procedure', '');
end
