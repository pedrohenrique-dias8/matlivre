if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Tests for linprog, quadprog and lsqlin. Prints:  RESULT name ef f x | it viol
o = optimoptions('linprog', 'Display', 'off');

[x, f, ef, out] = linprog([-1; 4], [-3 1; 1 2], [6; 4], [], [], [-inf; -3], [], o);
tqp_rep('LP1_scipy_doc', x, f, ef, out);
[x, f, ef, out, lam] = linprog([-3; -5], [1 0; 0 2; 3 2], [4; 12; 18], [], [], [0; 0], [], o);
tqp_rep('LP2_classic', x, f, ef, out);
fprintf('LAMBDA LP2 ineqlin %s\n', mat2str(lam.ineqlin.', 6));
[x, f, ef, out] = linprog([1; 1; 1], [], [], [1 1 1; 1 -1 0], [6; 1], [0; 0; 1], [10; 10; 1], o);
tqp_rep('LP3_eq_fixed', x, f, ef, out);
[x, f, ef] = linprog([1; 1], [1 1; -1 -1], [1; -3], [], [], [], [], o);
fprintf('RESULT LP4_infeasible ef=%d\n', ef);
[x, f, ef] = linprog([-1; 0], [1 -1], 1, [], [], [0; 0], [], o);
fprintf('RESULT LP5_unbounded ef=%d\n', ef);
% a 30x20 LP with a known optimum (built from a feasible point and dual certificate)
m = 30; n = 20;
Am = mod((1:m)' * (1:n) * 7, 11) / 11 - 0.4;
x_ok = mod((1:n)', 3) / 3;
bm = Am * x_ok + 0.5;
fm = -(mod((1:n)', 5) + 1) / 5;
tic; [x, f, ef, out] = linprog(fm, Am, bm, [], [], zeros(n, 1), ones(n, 1), o); t = toc;
tqp_rep('LP6_30x20', x(1:5), f, ef, out); fprintf('TIME LP6 %.4f\n', t);
writematrix(x.', 'out_lp6.csv');

q = optimoptions('quadprog', 'Display', 'off');
[x, f, ef, out, lam] = quadprog([1 -1; -1 2], [-2; -6], [1 1; -1 2; 2 1], [2; 2; 3], [], [], [0; 0], [], [], q);
tqp_rep('QP1_doc', x, f, ef, out);
fprintf('LAMBDA QP1 ineqlin %s\n', mat2str(lam.ineqlin.', 6));
n = 10;
M = mod((1:n)' * (1:n) * 3, 7) / 7 - 0.3;
Hq = M.' * M + 0.1 * eye(n);
fq = mod((1:n)', 4) - 1.5;
Aq = ones(1, n); bq = 2;
Aeqq = [1:n] / n; beqq = 0.5;
tic; [x, f, ef, out] = quadprog(Hq, fq, Aq, bq, Aeqq, beqq, -ones(n, 1), ones(n, 1), [], q); t = toc;
tqp_rep('QP2_10var', x(1:5), f, ef, out); fprintf('TIME QP2 %.4f\n', t);
writematrix(x.', 'out_qp2.csv');
[x, f, ef] = quadprog([1 0; 0 -1], [0; 0], [], [], [], [], [-1; -1], [1; 1], [], q);
fprintf('RESULT QP3_nonconvex ef=%d\n', ef);
prob = struct('H', eye(3), 'f', [-1; -2; -3], 'Aineq', [1 1 1], 'bineq', 3, 'options', q, 'solver', 'quadprog');
[x, f, ef, out] = quadprog(prob);
tqp_rep('QP4_struct', x, f, ef, out);

l = optimoptions('lsqlin', 'Display', 'off');
C = mod((1:8)' * (1:4) * 5, 9) / 9; d = (1:8)' / 4;
[x, rn, r, ef, out] = lsqlin(C, d, [], [], [], [], zeros(4, 1), 0.5 * ones(4, 1), [], l);
tqp_rep('LSQLIN1_bounds', x, rn, ef, out);
[x, rn, r, ef, out] = lsqlin(C, d, [1 1 1 1], 1, [1 -1 0 0], 0, [], [], [], l);
tqp_rep('LSQLIN2_lin', x, rn, ef, out);

function tqp_rep(name, x, f, ef, out)
fprintf('RESULT %s ef=%d f=%.10g x=%s | it=%d viol=%.1e\n', name, ef, f, mat2str(x(:).', 9), out.iterations, out.constrviolation);
end
