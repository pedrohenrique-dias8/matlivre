if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Tests for lsqnonlin, lsqcurvefit and fsolve. Prints RESULT lines.
o = optimoptions('lsqnonlin', 'Display', 'off');
% N1 Rosenbrock as least squares
[x, rn, r, ef, out] = lsqnonlin(@(x) [10 * (x(2) - x(1)^2); 1 - x(1)], [-1.2 1], [], [], o);
tnl_rep('N1_rosen_lsq', x, rn, ef, out);
% N2 same with a bound that cuts the minimum (x1 <= 0.5)
[x, rn, r, ef, out, lam] = lsqnonlin(@(x) [10 * (x(2) - x(1)^2); 1 - x(1)], [-1.2 1], [-inf -inf], [0.5 inf], o);
tnl_rep('N2_rosen_bound', x, rn, ef, out); fprintf('LAMBDA N2 upper %s\n', mat2str(lam.upper.', 6));
% N3 exponential decay fit (lsqcurvefit), noise-free data -> exact parameters
xd = linspace(0, 4, 25); yd = 2.5 * exp(-1.3 * xd) + 0.4;
[x, rn, r, ef, out] = lsqcurvefit(@(p, t) p(1) * exp(-p(2) * t) + p(3), [1 1 0], xd, yd, [], [], o);
tnl_rep('N3_expfit', x, rn, ef, out);
% N4 Lorentzian peak fit with user Jacobian and noisy (deterministic pseudo-noise) data
yn = 3 ./ (1 + ((xd - 2) / 0.5).^2) + 0.05 * sin(37 * xd);
[x, rn, r, ef, out] = lsqcurvefit(@tnl_lorentz, [1 1.5 1], xd, yn, [0 0 0.01], [10 4 5], optimoptions(o, 'SpecifyObjectiveGradient', true));
tnl_rep('N4_lorentz_jac', x, rn, ef, out);
writematrix(x, 'out_n4.csv');
% N5 Brown and Dennis (m = 20, n = 4): large-residual problem, f* = 85822.2016...
bd = @(x) arrayfun(@(i) (x(1) + x(2) * (i/5) - exp(i/5))^2 + (x(3) + x(4) * sin(i/5) - cos(i/5))^2, (1:20)');
[x, rn, r, ef, out] = lsqnonlin(bd, [25 5 -5 -1], [], [], optimoptions(o, 'MaxFunctionEvaluations', 5000, 'FunctionTolerance', 1e-12));
tnl_rep('N5_brown_dennis', x, rn, ef, out);

s = optimoptions('fsolve', 'Display', 'off');
% S1 MATLAB-doc-like 2x2 system
[x, fv, ef, out] = fsolve(@(x) [2*x(1) - x(2) - exp(-x(1)); -x(1) + 2*x(2) - exp(-x(2))], [-5 -5], s);
tnl_reps('S1_exp2', x, fv, ef, out);
% S2 Powell badly scaled
[x, fv, ef, out] = fsolve(@(x) [1e4 * x(1) * x(2) - 1; exp(-x(1)) + exp(-x(2)) - 1.0001], [0 1], optimoptions(s, 'MaxFunctionEvaluations', 2000));
tnl_reps('S2_powell_badscale', x, fv, ef, out);
% S3 matrix equation X*X*X = A (2x2), user Jacobian not given, matrix x0
A = [1 2; 3 4];
[x, fv, ef, out] = fsolve(@(X) X * X * X - A, ones(2), s);
tnl_reps('S3_matrix_cuberoot', x, fv, ef, out); fprintf('CHECK S3 %s\n', mat2str(x * x * x, 8));
% S4 no real root: x^2 + 1 = 0
[x, fv, ef] = fsolve(@(x) x^2 + 1, 2, s);
fprintf('RESULT S4_noroot ef=%d x=%.3g f=%.3g\n', ef, x, fv);
% S5 Broyden tridiagonal n = 10 with analytic Jacobian
[x, fv, ef, out] = fsolve(@tnl_broyden, -ones(10, 1), optimoptions(s, 'SpecifyObjectiveGradient', true));
tnl_reps('S5_broyden10_jac', x(1:4), fv, ef, out);

function tnl_rep(name, x, rn, ef, out)
fprintf('RESULT %s ef=%d resnorm=%.10g x=%s | it=%d fc=%d\n', name, ef, rn, mat2str(x(:).', 9), out.iterations, out.funcCount);
end
function tnl_reps(name, x, fv, ef, out)
fprintf('RESULT %s ef=%d |F|=%.2e x=%s | it=%d fc=%d\n', name, ef, norm(fv(:)), mat2str(x(:).', 9), out.iterations, out.funcCount);
end
function [F, J] = tnl_lorentz(p, t)
u = (t - p(2)) / p(3);
d = 1 + u.^2;
F = p(1) ./ d;
J = [1 ./ d(:), (p(1) * 2 * u(:) ./ d(:).^2) / p(3), (p(1) * 2 * u(:).^2 ./ d(:).^2) / p(3)];
end
function [F, J] = tnl_broyden(x)
n = numel(x); xm = [0; x(1:end-1)]; xp = [x(2:end); 0];
F = (3 - 2 * x) .* x - xm - 2 * xp + 1;
J = diag(3 - 4 * x) - diag(ones(n - 1, 1), -1) - 2 * diag(ones(n - 1, 1), 1);
end
