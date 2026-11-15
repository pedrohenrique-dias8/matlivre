function [x, fval, exitflag, output] = patternsearch(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
%PATTERNSEARCH  Derivative-free minimization by generalized pattern search.
%   x = patternsearch(fun, x0)
%   x = patternsearch(fun, x0, A, b)
%   x = patternsearch(fun, x0, A, b, Aeq, beq, lb, ub)
%   x = patternsearch(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon)
%   x = patternsearch(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
%   x = patternsearch(problem)
%   [x, fval, exitflag, output] = patternsearch(...)
%
%   GPS with the 2N positive basis (+-e_i): polls the mesh around the current point, moves to
%   the first better point (or the best, with UseCompletePoll), expands the mesh by
%   MeshExpansionFactor after a success and contracts it by MeshContractionFactor after a failure.
%   Points are compared by Deb's feasibility rules (feasible beats infeasible; infeasible points
%   are compared by their constraint violation), so the search can start from an infeasible x0.
%   Bounds are kept by clamping poll points; near linear constraints the poll set also includes
%   directions along the constraints (their null spaces) and the inward normals. MATLAB's patternsearch handles nonlinear
%   constraints with an augmented Lagrangian instead; results can differ on such problems.
%
%   Options: MeshTolerance (1e-6), StepTolerance (1e-6), FunctionTolerance (1e-6),
%   ConstraintTolerance (1e-6), MaxIterations (100*numel(x0)), MaxFunctionEvaluations
%   (2000*numel(x0)), InitialMeshSize (1), MeshExpansionFactor (2), MeshContractionFactor
%   (0.5), UseCompletePoll (false), Display ('final' | 'iter' | 'off').
%   exitflag: 1 mesh size below MeshTolerance, 2 step below StepTolerance and feasible,
%   0 limits reached, -2 no feasible point.
%
%   Reference: V. Torczon (1997), "On the convergence of pattern search algorithms"; C. Audet and
%   J. E. Dennis (2003). Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(fun)
    p = fun; fun = p.objective; x0 = p.x0;
    A = ps_fld(p, 'Aineq'); b = ps_fld(p, 'bineq'); Aeq = ps_fld(p, 'Aeq'); beq = ps_fld(p, 'beq');
    lb = ps_fld(p, 'lb'); ub = ps_fld(p, 'ub'); nonlcon = ps_fld(p, 'nonlcon'); options = ps_fld(p, 'options');
    nin = 10;
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
if nin < 10 || isempty(options), options = struct(); end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
shape = size(x0); x = double(x0(:)); n = numel(x);
if isempty(lb), lb = -inf(n, 1); end
if isempty(ub), ub = inf(n, 1); end
lb = double(lb(:)); ub = double(ub(:));
meshtol = optim_getopt(options, {'MeshTolerance', 'TolMesh'}, 1e-6);
tolx = optim_getopt(options, {'StepTolerance', 'TolX'}, 1e-6);
tolc = optim_getopt(options, {'ConstraintTolerance', 'TolCon'}, 1e-6);
maxit = optim_getopt(options, {'MaxIterations', 'MaxIter'}, 100 * n);
maxfev = optim_getopt(options, {'MaxFunctionEvaluations', 'MaxFunEvals'}, 2000 * n);
mesh = optim_getopt(options, {'InitialMeshSize'}, 1);
ex = optim_getopt(options, {'MeshExpansionFactor', 'MeshExpansion'}, 2);
ct = optim_getopt(options, {'MeshContractionFactor', 'MeshContraction'}, 0.5);
complete = optim_getopt(options, {'UseCompletePoll', 'CompletePoll'}, false);
if ischar(complete) || isstring(complete), complete = strcmpi(complete, 'on'); end
lvl = optim_display(options, 'final');

S = struct('fun', fun, 'nonlcon', nonlcon, 'A', A, 'b', b(:), 'Aeq', Aeq, 'beq', beq(:), 'shape', shape);
x = min(max(x, lb), ub);
[f, v] = ps_eval(S, x); nev = 1;
D = [eye(n), -eye(n)];
if lvl == 2
    fprintf('\nIter     Func-count       f(x)      MeshSize     Method\n');
    fprintf('%4d    %10d   %12.6g  %10.4g\n', 0, nev, f, mesh);
end
exitflag = 0; msg = sprintf('Stopped: MaxIterations = %d reached.', maxit); k = 0; lastk = 0;
OF = optim_outfcn('setup', options, 'ps');
userstop = false;
if OF.active
    [s1, OF] = optim_outfcn(OF, ps_ov(x, shape, 0, f, mesh, nev, '', v), options, 'init');
    [s2, OF] = optim_outfcn(OF, ps_ov(x, shape, 0, f, mesh, nev, '', v), options, 'iter');
    userstop = s1 || s2;
    if userstop, maxit = 0; end
end
for k = 1:maxit
    if nev >= maxfev
        msg = sprintf('Stopped: MaxFunctionEvaluations = %d reached.', maxfev); break
    end
    improved = false; xb = x; fb = f; vb = v;
    Dk = [D, ps_tangent(S, x, mesh)];
    for d = 1:size(Dk, 2)
        xt = min(max(x + mesh * Dk(:, d), lb), ub);
        if all(xt == x), continue, end
        [ft, vt] = ps_eval(S, xt); nev = nev + 1;
        if ps_better(ft, vt, fb, vb, tolc)
            xb = xt; fb = ft; vb = vt; improved = true;
            if ~complete, break, end
        end
    end
    if improved
        step = norm(xb - x);
        x = xb; f = fb; v = vb; mesh = mesh * ex; meth = 'Successful Poll';
        lastk = k;
    else
        mesh = mesh * ct; meth = 'Refine Mesh'; step = 0;
    end
    if lvl == 2
        fprintf('%4d    %10d   %12.6g  %10.4g     %s\n', k, nev, f, mesh, meth);
    end
    if OF.active
        [userstop, OF] = optim_outfcn(OF, ps_ov(x, shape, k, f, mesh, nev, meth, v), options, 'iter');
        if userstop, break, end
    end
    if mesh < meshtol
        exitflag = 1; msg = 'Optimization terminated: mesh size less than MeshTolerance.'; break
    end
    if improved && step < tolx * max(1, norm(x)) && mesh < sqrt(meshtol) && v <= tolc
        exitflag = 2; msg = 'Optimization terminated: change in X less than StepTolerance.'; break
    end
end
if v > tolc
    exitflag = -2; msg = 'No feasible point found: the constraint violation is above ConstraintTolerance.';
end
if userstop, exitflag = -1; msg = 'Optimization terminated: stop requested by an output function or plot function.'; end
if OF.active, optim_outfcn(OF, ps_ov(x, shape, k, f, mesh, nev, '', v), options, 'done'); end
x = reshape(x, shape); fval = f;
output = struct('function', fun, 'problemtype', '', 'pollmethod', 'gpspositivebasis2n', ...
    'iterations', k, 'funccount', nev, 'meshsize', mesh, 'maxconstraint', v, 'message', msg);
if lvl == 1 || lvl == 2 || (lvl == -1 && exitflag <= 0)
    fprintf('\n%s\n\n', msg);
end
end

function [f, v] = ps_eval(S, x)
X = reshape(x, S.shape);
f = double(S.fun(X)); if isnan(f), f = inf; end
v = 0;
if ~isempty(S.A), v = v + sum(max(0, S.A * x - S.b)); end
if ~isempty(S.Aeq), v = v + sum(abs(S.Aeq * x - S.beq)); end
if ~isempty(S.nonlcon)
    [c, ceq] = S.nonlcon(X);
    v = v + sum(max(0, c(:))) + sum(abs(ceq(:)));
end
end

function T = ps_tangent(S, x, mesh)
% extra poll directions along the linear constraints that are nearly active (within the mesh):
% the null space of each such constraint normal (both signs) and the inward normal, plus the
% null space of the equality constraints. Coordinate directions alone stall at the boundary of
% constraints that are not aligned with the axes.
n = numel(x); T = zeros(n, 0);
if ~isempty(S.A)
    r = S.A * x - S.b;
    nr = sqrt(sum(S.A.^2, 2));
    act = find(r >= -mesh * nr & nr > 0);
    for i = act(:).'
        a = S.A(i, :);
        N = null(a);
        T = [T, N, -N, -a.' / norm(a)]; %#ok<AGROW>
    end
end
if ~isempty(S.Aeq)
    N = null(S.Aeq);
    T = [T, N, -N];
end
end

function tf = ps_better(f1, v1, f2, v2, tolc)
if v1 <= tolc && v2 <= tolc, tf = f1 < f2;
elseif v1 <= tolc, tf = true;
elseif v2 <= tolc, tf = false;
else, tf = v1 < v2; end
end

function v = ps_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end

function ov = ps_ov(x, shape, k, f, mesh, nev, meth, v)
ov = struct('x', reshape(x, shape), 'iteration', k, 'fval', f, 'meshsize', mesh, ...
    'funccount', nev, 'method', meth, 'maxconstraint', v);
end
