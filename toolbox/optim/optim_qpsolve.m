function [x, exitflag, output, lambda] = optim_qpsolve(H, f, A, b, Aeq, beq, lb, ub, options, solver)
%OPTIM_QPSOLVE  Common front end of quadprog, linprog and lsqlin (MatLivre optimization toolbox).
%   Builds the constraint matrices from linear inequalities, equalities and bounds (fixed
%   variables, lb == ub, become equalities), calls the interior-point QP solver optim_qpip and
%   maps the result to MATLAB's exitflag, output and lambda conventions.
%
%   exitflag:  1 converged, 0 iteration limit, -2 infeasible, -3 unbounded, -6 nonconvex (QP)

f = double(f(:)); n = numel(f);
if isempty(H), H = zeros(n); end
if isempty(A), A = zeros(0, n); b = zeros(0, 1); end
if isempty(Aeq), Aeq = zeros(0, n); beq = zeros(0, 1); end
b = double(b(:)); beq = double(beq(:));
if size(A, 2) ~= n || size(A, 1) ~= numel(b)
    error(['MatLivre:' solver ':sizeA'], 'A must have %d columns and as many rows as b has elements.', n);
end
if size(Aeq, 2) ~= n || size(Aeq, 1) ~= numel(beq)
    error(['MatLivre:' solver ':sizeAeq'], 'Aeq must have %d columns and as many rows as beq has elements.', n);
end
if isempty(lb), lb = -inf(n, 1); end
if isempty(ub), ub = inf(n, 1); end
lb = double(lb(:)); ub = double(ub(:));
if numel(lb) < n, lb = [lb; -inf(n - numel(lb), 1)]; end
if numel(ub) < n, ub = [ub; inf(n - numel(ub), 1)]; end
lb = lb(1:n); ub = ub(1:n);

maxit = optim_getopt(options, {'MaxIterations', 'MaxIter'}, 200);
tolcon = optim_getopt(options, {'ConstraintTolerance', 'TolCon'}, 1e-8);
tolopt = optim_getopt(options, {'OptimalityTolerance', 'TolFun'}, 1e-8);
lvl = optim_display(options, 'final');

if strcmp(solver, 'linprog'), alg = 'interior-point'; else, alg = 'interior-point-convex'; end
output = struct('iterations', 0, 'constrviolation', NaN, 'firstorderopt', NaN, ...
    'algorithm', alg, 'message', '');
lambda = struct('ineqlin', zeros(size(A, 1), 1), 'eqlin', zeros(size(Aeq, 1), 1), ...
    'lower', zeros(n, 1), 'upper', zeros(n, 1));
x = [];

if any(lb > ub)
    exitflag = -2; output.message = 'No feasible point: lb(i) > ub(i) for some i.';
    qps_show(lvl, exitflag, output.message);
    return
end
H = (double(H) + double(H).') / 2;
if any(H(:) ~= 0)
    ev = eig(H);
    if min(ev) < -1e-10 * max(1, max(abs(ev)))
        exitflag = -6; output.message = 'Nonconvex problem: H is not positive semidefinite (this algorithm needs a convex problem).';
        qps_show(lvl, exitflag, output.message);
        return
    end
end

fix = find(lb == ub);
ilb = find(isfinite(lb) & lb ~= ub); iub = find(isfinite(ub) & lb ~= ub);
E = eye(n);
Ai = [A; -E(ilb, :); E(iub, :)];
bi = [b; -lb(ilb); ub(iub)];
Ae = [Aeq; E(fix, :)];
be = [beq; lb(fix)];

[x, y, z, info] = optim_qpip(H, f, Ai, bi, Ae, be, struct('tol', 1e-10, 'maxit', maxit));
mA = size(A, 1); nl = numel(ilb);
lambda.ineqlin = z(1:mA);
lambda.lower(ilb) = z(mA + 1:mA + nl);
lambda.upper(iub) = z(mA + nl + 1:end);
lambda.eqlin = y(1:size(Aeq, 1));
yfix = y(size(Aeq, 1) + 1:end);           % multiplier of x(fix) = lb(fix)
lambda.lower(fix) = max(-yfix, 0);
lambda.upper(fix) = max(yfix, 0);

viol = max([0; A * x - b; abs(Aeq * x - beq); lb - x; x - ub]);
gl = H * x + f + Ai.' * z + Ae.' * y;
output.iterations = info.iterations;
output.constrviolation = viol;
output.firstorderopt = max([norm(gl, inf); abs(z .* (Ai * x - bi)); 0]);
scale = 1 + max([norm(f, inf), norm(H, inf), norm(b, inf), norm(beq, inf)]);
if info.converged || (viol <= tolcon * (1 + norm(x, inf)) && output.firstorderopt <= tolopt * scale)
    exitflag = 1; output.message = 'Minimum found that satisfies the constraints.';
elseif info.diverged && viol <= sqrt(tolcon) * (1 + norm(x, inf))
    exitflag = -3; output.message = 'Problem is unbounded: the objective decreases without limit along a feasible direction.';
elseif info.primal > sqrt(tolcon) * scale
    exitflag = -2; output.message = 'No feasible point found: the constraints cannot be satisfied.';
elseif info.diverged
    exitflag = -3; output.message = 'Problem appears unbounded.';
else
    exitflag = 0; output.message = sprintf('Stopped: the number of iterations reached MaxIterations = %d.', maxit);
end
qps_show(lvl, exitflag, output.message);
end

function qps_show(lvl, ef, msg)
if lvl == 1 || lvl == 2 || (lvl == -1 && ef <= 0)
    fprintf('\n%s\n\n', msg);
end
end
