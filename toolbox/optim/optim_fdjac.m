function [J, nev] = optim_fdjac(fun, x, fx, shape, type, step, lb, ub)
%OPTIM_FDJAC  Finite-difference Jacobian of a vector function (MatLivre optimization toolbox).
%   [J, nev] = optim_fdjac(fun, x, fx, shape, type, step, lb, ub)
%   fun   handle called as fun(reshape(x, shape)), returning a vector (or a scalar)
%   x     column vector; fx = fun at x, as a column
%   type  'forward' or 'central'; step = relative step (scalar or vector)
%   lb, ub optional bounds: steps are flipped or made one-sided so fun is never called outside
%   Returns J (numel(fx)-by-numel(x)) and the number of function evaluations used.
n = numel(x); m = numel(fx);
if nargin < 7 || isempty(lb), lb = -inf(n, 1); end
if nargin < 8 || isempty(ub), ub = inf(n, 1); end
J = zeros(m, n); nev = 0;
central = strcmpi(type, 'central');
for j = 1:n
    if numel(step) > 1, hj = step(j); else, hj = step; end
    h = hj * max(abs(x(j)), 1);
    if central && x(j) + h <= ub(j) && x(j) - h >= lb(j)
        xp = x; xp(j) = x(j) + h; xm = x; xm(j) = x(j) - h;
        fp = fun(reshape(xp, shape)); fm = fun(reshape(xm, shape)); nev = nev + 2;
        J(:, j) = (fp(:) - fm(:)) / (2 * h);
    else
        if x(j) + h > ub(j), h = -h; end
        xp = x; xp(j) = x(j) + h;
        fp = fun(reshape(xp, shape)); nev = nev + 1;
        J(:, j) = (fp(:) - fx(:)) / h;
    end
end
end
