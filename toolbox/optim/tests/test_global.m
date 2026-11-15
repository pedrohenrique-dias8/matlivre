if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Tests for the global solvers. Stochastic solvers are judged by their success rate over 20
% seeds (rng(1) ... rng(20)), since MatLivre's random streams differ from MATLAB's.
ras = @(x) 10 * numel(x) + sum(x.^2 - 10 * cos(2 * pi * x));
ack = @(x) -20 * exp(-0.2 * sqrt(mean(x.^2))) - exp(mean(cos(2 * pi * x))) + 20 + exp(1);
po = optimoptions('particleswarm', 'Display', 'off');
go = optimoptions('ga', 'Display', 'off');
cases = {{'rastrigin2', ras, 2, 5.12}, {'rastrigin5', ras, 5, 5.12}, {'ackley10', ack, 10, 32}};
for c = 1:numel(cases)
    cc = cases{c};
    [okp, mp, evp, tp] = trial(@(s) particleswarm(cc{2}, cc{3}, -cc{4} * ones(1, cc{3}), cc{4} * ones(1, cc{3}), po), 1e-3);
    [okg, mg, evg, tg] = trial(@(s) ga(cc{2}, cc{3}, [], [], [], [], -cc{4} * ones(1, cc{3}), cc{4} * ones(1, cc{3}), [], go), 1e-3);
    fprintf('GLOBAL %-11s particleswarm %2d/20 (median f %.3g, %5d evals, %.2fs) | ga %2d/20 (median f %.3g, %5d evals, %.2fs)\n', ...
        cc{1}, okp, mp, evp, tp, okg, mg, evg, tg);
end
% hybrid
rng(1); [x, f, ef, out] = particleswarm(ras, 2, [-5.12 -5.12], [5.12 5.12], optimoptions(po, 'HybridFcn', 'fmincon'));
fprintf('RESULT pso_hybrid f=%.3g ef=%d hybridflag=%d\n', f, ef, out.hybridflag);
% ga with constraints, integers, linear equality
ok = 0;
for s = 1:20
    rng(s); [x, f, ef] = ga(@(x) (x(1) - 2)^2 + (x(2) - 1)^2, 2, [1 1], 2, [], [], [-5 -5], [5 5], @(x) deal(x(1)^2 - x(2), []), go);
    ok = ok + (abs(f - 1) < 1e-2 && ef >= 0);
end
fprintf('RESULT ga_nonlcon success %d/20\n', ok);
rng(1); [x, f, ef] = ga(@(x) (x(1) - 2)^2 + (x(2) - 1)^2, 2, [1 1], 2, [], [], [-5 -5], [5 5], @(x) deal(x(1)^2 - x(2), []), optimoptions(go, 'HybridFcn', 'fmincon'));
fprintf('RESULT ga_nonlcon_hybrid f=%.8f x=%s ef=%d\n', f, mat2str(x, 6), ef);
ok = 0;
for s = 1:20
    rng(s); x = ga(@(x) (x(1) - 0.4)^2 + (x(2) - 2.6)^2 + (x(3) + 1.7)^2, 3, [], [], [], [], -5 * ones(1, 3), 5 * ones(1, 3), [], 1:3, go);
    ok = ok + isequal(x, [0 3 -2]);
end
fprintf('RESULT ga_integer success %d/20\n', ok);
rng(1); [x, f, ef, out] = ga(@(x) sum((x - [1.3 2.7 0.2]).^2), 3, [], [], [1 1 1], 4, -5 * ones(1, 3), 5 * ones(1, 3), [], 2, go);
fprintf('RESULT ga_mixedint_eq x=%s f=%.4f ef=%d (optimum x=[1.05 3 -0.05], f=0.2150)\n', mat2str(x, 4), f, ef);
% patternsearch
pso = optimoptions('patternsearch', 'Display', 'off');
[x, f, ef, out] = patternsearch(@(x) 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2, [-1.2 1], [], [], [], [], [], [], [], optimoptions(pso, 'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000));
fprintf('RESULT ps_rosen f=%.3g x=%s ef=%d fc=%d\n', f, mat2str(x, 5), ef, out.funccount);
[x, f, ef, out] = patternsearch(@(x) (x(1) - 1)^2 + (x(2) - 2.5)^2, [2 0], [-1 2; 1 2; 1 -2], [2; 6; 2], [], [], [0 0], [], [], pso);
fprintf('RESULT ps_linear f=%.6f x=%s ef=%d (optimum x=[1.4 1.7], f=0.8)\n', f, mat2str(x, 6), ef);
[x, f, ef, out] = patternsearch(@(x) x(1) + x(2), [3 3], [], [], [], [], [], [], @(x) deal(x(1)^2 + x(2)^2 - 2, []), pso);
fprintf('RESULT ps_nonlcon f=%.6f x=%s ef=%d (optimum x=[-1 -1], f=-2)\n', f, mat2str(x, 6), ef);
% MultiStart / GlobalSearch on the six-hump camel (two global minima, f* = -1.0316)
camel = @(x) (4 - 2.1 * x(1)^2 + x(1)^4 / 3) * x(1)^2 + x(1) * x(2) + (-4 + 4 * x(2)^2) * x(2)^2;
prob = createOptimProblem('fmincon', 'objective', camel, 'x0', [2 2], 'lb', [-3 -2], 'ub', [3 2], ...
    'options', optimoptions('fmincon', 'Display', 'off'));
rng(1); [x, f, ef, out, sols] = run(MultiStart('Display', 'off'), prob, 30);
fprintf('RESULT multistart f=%.6f ef=%d runs=%d distinct minima=%d\n', f, ef, out.localSolverTotal, numel(sols));
rng(1); [x, f, ef, out, sols] = run(GlobalSearch('Display', 'off'), prob);
fprintf('RESULT globalsearch f=%.6f ef=%d runs=%d distinct minima=%d\n', f, ef, out.localSolverTotal, numel(sols));
% MultiStart with lsqcurvefit: fit a sum of two exponentials (many local minima)
xd = linspace(0, 5, 40); yd = 3 * exp(-0.5 * xd) + 1.5 * exp(-3 * xd);
pc = createOptimProblem('lsqcurvefit', 'objective', @(p, t) p(1) * exp(-p(2) * t) + p(3) * exp(-p(4) * t), ...
    'x0', [1 1 1 1], 'xdata', xd, 'ydata', yd, 'lb', [0 0 0 0], 'ub', [10 10 10 10], 'options', optimoptions('lsqcurvefit', 'Display', 'off'));
rng(2); [x, f, ef] = run(MultiStart('Display', 'off'), pc, 15);
fprintf('RESULT multistart_lsqcurvefit resnorm=%.3g x=%s ef=%d\n', f, mat2str(sortrows(reshape(x, 2, 2).').', 4), ef);

function [ok, med, ev, t] = trial(solve, tol)
fs = zeros(1, 20); ev = 0; tic;
for s = 1:20
    rng(s);
    [~, f, ~, out] = solve(s);
    fs(s) = f; ev = ev + out.funccount;
end
t = toc; ok = sum(fs < tol); med = median(fs); ev = round(ev / 20);
end
