function [x, resnorm, residual, exitflag, output, lambda] = lsqlin(C, d, A, b, Aeq, beq, lb, ub, x0, options)
%LSQLIN  Constrained linear least squares.
%   x = lsqlin(C, d, A, b)                      minimizes ||C*x - d||^2 subject to A*x <= b
%   x = lsqlin(C, d, A, b, Aeq, beq, lb, ub)    and Aeq*x = beq, lb <= x <= ub
%   x = lsqlin(C, d, A, b, Aeq, beq, lb, ub, x0, options)
%   x = lsqlin(problem)
%   [x, resnorm, residual, exitflag, output, lambda] = lsqlin(...)
%
%   resnorm = ||C*x - d||^2, residual = C*x - d. Solved as the convex QP
%   0.5*x'*(C'*C)*x - (C'*d)'*x by the interior-point solver optim_qpip.
%   Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(C)
    p = C;
    C = lsl_fld(p, 'C'); d = lsl_fld(p, 'd'); A = lsl_fld(p, 'Aineq'); b = lsl_fld(p, 'bineq');
    Aeq = lsl_fld(p, 'Aeq'); beq = lsl_fld(p, 'beq'); lb = lsl_fld(p, 'lb'); ub = lsl_fld(p, 'ub');
    options = lsl_fld(p, 'options');
else
    if nargin < 3, A = []; end
    if nargin < 4, b = []; end
    if nargin < 5, Aeq = []; end
    if nargin < 6, beq = []; end
    if nargin < 7, lb = []; end
    if nargin < 8, ub = []; end
    if nargin < 10, options = struct(); end
end
C = double(C); d = double(d(:));
H = C.' * C; f = -(C.' * d);
[x, exitflag, output, lambda] = optim_qpsolve(H, f, A, b, Aeq, beq, lb, ub, options, 'lsqlin');
if isempty(x)
    resnorm = []; residual = [];
else
    residual = C * x - d;
    resnorm = residual.' * residual;
end
end

function v = lsl_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end
