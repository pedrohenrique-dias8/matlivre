function [valid, err] = checkGradients(fun, x0, varargin)
%CHECKGRADIENTS  Compare supplied derivatives with central finite differences.
%   valid = checkGradients(fun, x0)
%   valid = checkGradients(fun, x0, options)
%   valid = checkGradients(..., 'IsConstraint', tf, 'Display', 'on'|'off', 'Tolerance', tol)
%   [valid, err] = checkGradients(...)
%
%   Objective (IsConstraint false): fun returns [f, g] with g the gradient (numel(x0) elements),
%   or [F, J] with F a vector and J its Jacobian (numel(F)-by-numel(x0)), as used by fminunc,
%   fmincon, lsqnonlin, lsqcurvefit and fsolve. valid is true when every element of the relative
%   difference is at most Tolerance; err.Objective holds the relative differences.
%
%   Constraints (IsConstraint true): fun is a nonlcon function returning [c, ceq, gc, gceq] with
%   gradients in columns (numel(x0)-by-numel(c)), as for fmincon. valid = [validIneq validEq] and
%   err has the fields Inequality and Equality.
%
%   Relative difference: abs(supplied - fd) ./ max(abs(supplied), 1), element by element, with fd
%   the finite-difference derivative (options.FiniteDifferenceType, 'forward' by default, step
%   options.FiniteDifferenceStepSize, default sqrt(eps) forward or eps^(1/3) central, scaled by
%   max(abs(x), 1)). Default Tolerance 1e-6, Display 'off'.
%   Part of the MatLivre optimization toolbox draft (clean-room).
options = struct(); k = 1;
if ~isempty(varargin) && (isstruct(varargin{1}) || isempty(varargin{1}))
    if isstruct(varargin{1}), options = varargin{1}; end
    k = 2;
end
iscon = false; disp_on = false; tol = 1e-6;
if mod(numel(varargin) - k + 1, 2) ~= 0
    error('MatLivre:checkGradients:pairs', 'Optional arguments must be name-value pairs.');
end
for i = k:2:numel(varargin)
    name = char(varargin{i}); val = varargin{i + 1};
    switch lower(name)
        case 'isconstraint', iscon = logical(val);
        case 'display', disp_on = strcmpi(char(val), 'on');
        case 'tolerance', tol = val;
        otherwise, error('MatLivre:checkGradients:name', 'Unknown option ''%s''.', name);
    end
end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
fdtype = lower(char(optim_getopt(options, {'FiniteDifferenceType'}, 'forward')));
if strcmp(fdtype, 'central'), defstep = eps^(1/3); else, defstep = sqrt(eps); end
step = optim_getopt(options, {'FiniteDifferenceStepSize'}, defstep);
shape = size(x0); x = double(x0(:)); n = numel(x);

if ~iscon
    [f, G] = fun(x0);
    f = double(f(:));
    if numel(f) == 1
        Gs = reshape(double(G), 1, []);          % gradient as a row (1-by-n)
    else
        Gs = double(G);                          % Jacobian m-by-n
    end
    Gfd = cg_fd(@(z) cg_val1(fun, z), x, shape, step, numel(f), fdtype, f);
    cg_size(Gs, Gfd, 'objective');
    E = abs(Gs - Gfd) ./ max(abs(Gs), 1);
    if numel(f) == 1, E = E.'; end
    valid = all(E(:) <= tol);
    err = struct('Objective', E);
    if disp_on, cg_report('Objective function derivatives', E, tol); end
else
    [c, ceq, gc, gceq] = fun(x0);
    mc = numel(c); me = numel(ceq);
    Jc = cg_fd(@(z) cg_val_c(fun, z), x, shape, step, mc, fdtype, double(c(:)));
    Je = cg_fd(@(z) cg_val_ceq(fun, z), x, shape, step, me, fdtype, double(ceq(:)));
    if mc > 0
        cg_size(double(gc).', Jc, 'inequality constraint');
        Ei = (abs(double(gc).' - Jc) ./ max(abs(double(gc).'), 1)).';
    else
        Ei = zeros(n, 0);
    end
    if me > 0
        cg_size(double(gceq).', Je, 'equality constraint');
        Ee = (abs(double(gceq).' - Je) ./ max(abs(double(gceq).'), 1)).';
    else
        Ee = zeros(n, 0);
    end
    valid = [all(Ei(:) <= tol), all(Ee(:) <= tol)];
    err = struct('Inequality', Ei, 'Equality', Ee);
    if disp_on
        if mc > 0, cg_report('Nonlinear inequality constraint derivatives', Ei, tol); end
        if me > 0, cg_report('Nonlinear equality constraint derivatives', Ee, tol); end
    end
end
end

function v = cg_val1(fun, z)
v = fun(z); v = double(v(:));
end

function v = cg_val_c(fun, z)
[v, ~] = fun(z); v = double(v(:));
end

function v = cg_val_ceq(fun, z)
[~, v] = fun(z); v = double(v(:));
end

function J = cg_fd(f, x, shape, step, m, fdtype, f0)
n = numel(x); J = zeros(m, n);
if m == 0, return, end
for j = 1:n
    if numel(step) > 1, hj = step(j); else, hj = step; end
    h = hj * max(abs(x(j)), 1);
    xp = x; xp(j) = x(j) + h;
    if strcmp(fdtype, 'central')
        xm = x; xm(j) = x(j) - h;
        J(:, j) = (f(reshape(xp, shape)) - f(reshape(xm, shape))) / (2 * h);
    else
        J(:, j) = (f(reshape(xp, shape)) - f0) / h;
    end
end
end

function cg_size(Gs, Gfd, what)
if ~isequal(size(Gs), size(Gfd))
    error('MatLivre:checkGradients:size', ...
        'The supplied %s derivative has size %s; the finite-difference derivative has size %s.', ...
        what, mat2str(size(Gs)), mat2str(size(Gfd)));
end
end

function cg_report(label, E, tol)
[mx, i] = max(E(:));
if isempty(mx), return, end
[r, c] = ind2sub(size(E), i);
fprintf('\n%s:\n  Maximum relative difference between supplied and finite-difference derivatives = %g.\n', label, mx);
if mx <= tol
    fprintf('  checkGradients succeeded: every relative difference is within Tolerance = %g.\n', tol);
else
    fprintf('  Largest difference at element (%d,%d). checkGradients failed: Tolerance = %g.\n', r, c, tol);
end
end
