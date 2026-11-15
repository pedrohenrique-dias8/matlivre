function [x, fval, exitflag, output, lambda] = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options)
%QUADPROG  Quadratic programming.
%   x = quadprog(H, f)                         minimizes 0.5*x'*H*x + f'*x
%   x = quadprog(H, f, A, b)                   subject to A*x <= b
%   x = quadprog(H, f, A, b, Aeq, beq)         and Aeq*x = beq
%   x = quadprog(H, f, A, b, Aeq, beq, lb, ub) and lb <= x <= ub
%   x = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options)
%   x = quadprog(problem)
%   [x, fval, exitflag, output, lambda] = quadprog(...)
%
%   Convex problems only (H positive semidefinite), solved by a primal-dual interior-point method
%   (optim_qpip). x0 is accepted and ignored, as in MATLAB's interior-point-convex algorithm.
%   exitflag: 1 solved, 0 iteration limit, -2 infeasible, -3 unbounded, -6 nonconvex.
%   Options: Display, MaxIterations, OptimalityTolerance, ConstraintTolerance.
%   Part of the MatLivre optimization toolbox draft (clean-room; Nocedal & Wright, Sec. 16.6).
if nargin == 1 && isstruct(H)
    p = H;
    H = qpg_fld(p, 'H'); f = qpg_fld(p, 'f'); A = qpg_fld(p, 'Aineq'); b = qpg_fld(p, 'bineq');
    Aeq = qpg_fld(p, 'Aeq'); beq = qpg_fld(p, 'beq'); lb = qpg_fld(p, 'lb'); ub = qpg_fld(p, 'ub');
    options = qpg_fld(p, 'options');
else
    if nargin < 2, error('MatLivre:quadprog:notEnoughInputs', 'quadprog requires at least two input arguments.'); end
    if nargin < 3, A = []; end
    if nargin < 4, b = []; end
    if nargin < 5, Aeq = []; end
    if nargin < 6, beq = []; end
    if nargin < 7, lb = []; end
    if nargin < 8, ub = []; end
    if nargin < 10, options = struct(); end
end
if isempty(f), f = zeros(size(H, 1), 1); end
[x, exitflag, output, lambda] = optim_qpsolve(H, f, A, b, Aeq, beq, lb, ub, options, 'quadprog');
if isempty(x)
    fval = [];
else
    fval = 0.5 * x.' * ((H + H.') / 2) * x + f(:).' * x;
end
end

function v = qpg_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end
