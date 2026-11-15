if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Issue #8 use case: DC circuit, 12 V source, Rc and Rh in series.
% Maximize power on Rh subject to I <= 5 A.
% Analytical optimum: Rc = 0.01, Rh = 2.39 (I = 5 A exactly), P = 59.75 W.

V = 12; Imax = 5;
lb = [0.01 0.01]; ub = [10 10];
obj = @(x) -V^2*x(2)/(x(1)+x(2))^2;
nonlcon = @(x) deal(V/(x(1)+x(2)) - Imax, []);
penalty = @(x) 1e8*max(0, V/(x(1)+x(2)) - Imax)^2;
psoObj = @(x) obj(x) + penalty(x);

optsGA = optimoptions('ga', 'PopulationSize', 100, 'MaxGenerations', 200, ...
    'FunctionTolerance', 1e-10, 'Display', 'off');
optsPSO = optimoptions('particleswarm', 'SwarmSize', 100, 'MaxIterations', 200, ...
    'FunctionTolerance', 1e-10, 'Display', 'off');

xs = [0.01 2.39]; Pst = 59.75;
nrun = 10;
fprintf('Reference optimum: Rc = %.4f, Rh = %.4f, P = %.4f W, I = %.4f A\n\n', xs(1), xs(2), Pst, V/sum(xs));

fprintf('%-4s %-5s %9s %9s %10s %9s %5s %7s %8s\n', 'solv', 'seed', 'Rc', 'Rh', 'P [W]', 'I [A]', 'flag', 'iters', 'time[s]');
names = {'ga', 'pso'};
for s = 1:2
    Ps = zeros(1, nrun); ts = zeros(1, nrun);
    for k = 1:nrun
        rng(k);
        t0 = tic;
        if s == 1
            [x, fval, flag, out] = ga(obj, 2, [], [], [], [], lb, ub, nonlcon, optsGA);
            it = out.generations;
        else
            [x, fval, flag, out] = particleswarm(psoObj, 2, lb, ub, optsPSO);
            it = out.iterations;
            fval = obj(x);
        end
        ts(k) = toc(t0);
        Ps(k) = -fval;
        fprintf('%-4s %-5d %9.5f %9.5f %10.5f %9.5f %5d %7d %8.3f\n', names{s}, k, x(1), x(2), -fval, V/sum(x), flag, it, ts(k));
    end
    fprintf('%s: P mean %.5f, worst %.5f, best %.5f, gap to 59.75 (worst) %.2e, mean time %.3f s\n\n', ...
        names{s}, mean(Ps), min(Ps), max(Ps), Pst - min(Ps), mean(ts));
end
