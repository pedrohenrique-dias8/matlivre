function [x, fval, exitflag, output, lambda] = linprog(f, A, b, Aeq, beq, lb, ub, options)
%LINPROG  Linear programming.
%   x = linprog(f, A, b)                       minimizes f'*x subject to A*x <= b
%   x = linprog(f, A, b, Aeq, beq)             and Aeq*x = beq
%   x = linprog(f, A, b, Aeq, beq, lb, ub)     and lb <= x <= ub
%   x = linprog(f, A, b, Aeq, beq, lb, ub, options)
%   x = linprog(problem)
%   [x, fval, exitflag, output, lambda] = linprog(...)
%
%   Algorithm 'dual-simplex' (default): bounded dual simplex (optim_simplex); the solution is a
%   vertex. Algorithm 'interior-point': primal-dual interior point (optim_qpip); when the
%   optimum is not unique the point lies inside the optimal face.
%   exitflag: 1 solved, 0 iteration limit, -2 infeasible, -3 unbounded.
%   lambda: ineqlin, eqlin, lower, upper (Lagrangian f'*x + ineqlin'*(A*x - b) +
%   eqlin'*(Aeq*x - beq) - lower'*(x - lb) + upper'*(x - ub); ineqlin, lower, upper >= 0).
%   Part of the MatLivre optimization toolbox draft (clean-room; Nocedal & Wright, Ch. 13-14).
if nargin == 1 && isstruct(f)
    p = f;
    f = lpg_fld(p, 'f'); A = lpg_fld(p, 'Aineq'); b = lpg_fld(p, 'bineq'); Aeq = lpg_fld(p, 'Aeq');
    beq = lpg_fld(p, 'beq'); lb = lpg_fld(p, 'lb'); ub = lpg_fld(p, 'ub'); options = lpg_fld(p, 'options');
    nin = 8;
else
    nin = nargin;
end
if nin < 2, A = []; end
if nin < 3, b = []; end
if nin < 4, Aeq = []; end
if nin < 5, beq = []; end
if nin < 6, lb = []; end
if nin < 7, ub = []; end
if nin < 8 || isempty(options), options = struct(); end
n = numel(f);
alg = lower(char(optim_getopt(options, {'Algorithm'}, 'dual-simplex')));
if any(strcmp(alg, {'interior-point', 'interior-point-legacy'}))
    [x, exitflag, output, lambda] = optim_qpsolve(zeros(n), f, A, b, Aeq, beq, lb, ub, options, 'linprog');
    if isempty(x), fval = []; else, fval = f(:).' * x; end
    return
end

% ---- dual simplex
f = double(f(:));
if isempty(A), A = zeros(0, n); b = zeros(0, 1); end
if isempty(Aeq), Aeq = zeros(0, n); beq = zeros(0, 1); end
b = double(b(:)); beq = double(beq(:));
if size(A, 2) ~= n || size(A, 1) ~= numel(b)
    error('MatLivre:linprog:sizeA', 'A must have %d columns and as many rows as b has elements.', n);
end
if size(Aeq, 2) ~= n || size(Aeq, 1) ~= numel(beq)
    error('MatLivre:linprog:sizeAeq', 'Aeq must have %d columns and as many rows as beq has elements.', n);
end
if isempty(lb), lb = -inf(n, 1); end
if isempty(ub), ub = inf(n, 1); end
lb = double(lb(:)); ub = double(ub(:));
if numel(lb) < n, lb = [lb; -inf(n - numel(lb), 1)]; end
if numel(ub) < n, ub = [ub; inf(n - numel(ub), 1)]; end
lb = lb(1:n); ub = ub(1:n);
lvl = optim_display(options, 'final');
maxit = optim_getopt(options, {'MaxIterations', 'MaxIter'}, 10 * (n + size(A, 1) + size(Aeq, 1)) + 1000);
output = struct('iterations', 0, 'constrviolation', NaN, 'firstorderopt', NaN, ...
    'algorithm', 'dual-simplex', 'message', '');
lambda = struct('ineqlin', zeros(size(A, 1), 1), 'eqlin', zeros(size(Aeq, 1), 1), ...
    'lower', zeros(n, 1), 'upper', zeros(n, 1));
x = []; fval = [];
if any(lb > ub)
    exitflag = -2; output.message = 'No feasible point: lb(i) > ub(i) for some i.';
    lpg_show(lvl, exitflag, output.message); return
end
[xs, st, info] = optim_simplex(f, A, b, Aeq, beq, lb, ub, struct('maxit', maxit));
output.iterations = info.iterations;
switch st
    case 1
        exitflag = 1; output.message = 'Optimal solution found.';
    case -2
        exitflag = -2; output.message = 'No feasible point found: the constraints cannot be satisfied.';
    case -3
        exitflag = -3; output.message = 'Problem is unbounded: the objective decreases without limit along a feasible direction.';
    otherwise
        exitflag = 0; output.message = sprintf('Stopped: the number of iterations reached MaxIterations = %d.', maxit);
end
if st == 1 || st == 0
    x = xs; fval = f.' * x;
    mA = size(A, 1);
    lambda.ineqlin = max(-info.y(1:mA), 0);
    lambda.eqlin = -info.y(mA + 1:end);
    d = info.d(1:n); stat = info.basis.stat(1:n);
    lambda.lower(stat == -1) = max(d(stat == -1), 0);
    lambda.upper(stat == 1) = max(-d(stat == 1), 0);
    output.constrviolation = max([0; A * x - b; abs(Aeq * x - beq); lb - x; x - ub]);
    gl = f + A.' * lambda.ineqlin + Aeq.' * lambda.eqlin - lambda.lower + lambda.upper;
    output.firstorderopt = norm(gl, inf);
end
lpg_show(lvl, exitflag, output.message);
end

function lpg_show(lvl, ef, msg)
if lvl == 1 || lvl == 2 || (lvl == -1 && ef <= 0)
    fprintf('\n%s\n\n', msg);
end
end

function v = lpg_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end
