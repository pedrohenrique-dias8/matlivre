if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Tests for fminunc. Prints:  RESULT name ef f |x-x*| it fc
o = optimoptions('fminunc', 'Display', 'off');
rosen = @(x) sum(100 * (x(2:end) - x(1:end-1).^2).^2 + (1 - x(1:end-1)).^2);
[x, f, ef, out] = fminunc(rosen, [-1.2 1], o);                 tfu_rep('U1_rosen2', x, ones(1, 2), f, ef, out);
[x, f, ef, out] = fminunc(rosen, -ones(1, 10), optimoptions(o, 'MaxIterations', 2000, 'MaxFunctionEvaluations', 1e5));
tfu_rep('U2_rosen10', x, ones(1, 10), f, ef, out);
[x, f, ef, out] = fminunc(@tfu_rosen_fg, [-1.2 1], optimoptions(o, 'SpecifyObjectiveGradient', true));
tfu_rep('U3_rosen2_grad', x, ones(1, 2), f, ef, out);
beale = @(x) (1.5 - x(1) + x(1)*x(2))^2 + (2.25 - x(1) + x(1)*x(2)^2)^2 + (2.625 - x(1) + x(1)*x(2)^3)^2;
[x, f, ef, out] = fminunc(beale, [1 1], o);                    tfu_rep('U4_beale', x, [3 0.5], f, ef, out);
powell = @(x) (x(1) + 10*x(2))^2 + 5*(x(3) - x(4))^2 + (x(2) - 2*x(3))^4 + 10*(x(1) - x(4))^4;
[x, f, ef, out] = fminunc(powell, [3 -1 0 1], o);              tfu_rep('U5_powell_singular', x, zeros(1, 4), f, ef, out);
D = diag(logspace(0, 4, 6));
[x, f, ef, out] = fminunc(@(x) 0.5 * x(:).' * D * x(:) - sum(x(:)), ones(2, 3), optimoptions(o, 'FiniteDifferenceType', 'central'));
tfu_rep('U6_illcond_matrix', x, 1 ./ diag(D).', f, ef, out); fprintf('SHAPE U6 %s\n', mat2str(size(x)));
[x, f, ef, out] = fminunc(@(x) x(1) + x(2)^2, [0 1], o);
fprintf('RESULT U7_unbounded ef=%d f=%.3g\n', ef, f);

function tfu_rep(name, x, xs, f, ef, out)
fprintf('RESULT %s ef=%d f=%.6e err=%.2e it=%d fc=%d\n', name, ef, f, norm(x(:) - xs(:), inf), out.iterations, out.funcCount);
end

function [f, g] = tfu_rosen_fg(x)
f = 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
g = [-400 * x(1) * (x(2) - x(1)^2) - 2 * (1 - x(1)); 200 * (x(2) - x(1)^2)];
end
