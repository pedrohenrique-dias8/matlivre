if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Tests for intlinprog. Prints MILP lines: name exitflag fval x | nodes
o = optimoptions('intlinprog', 'Display', 'off');
% M1 knapsack 0/1 (12 items)
w = [23 31 29 44 53 38 63 85 89 82 37 52]; v = [92 57 49 68 60 43 67 84 87 72 41 55];
tic; [x, f, ef, out] = intlinprog(-v, 1:12, w, 165, [], [], zeros(12, 1), ones(12, 1), [], o); t = toc;
rep('M1_knapsack12', x, f, ef, out, t);
% M2 MATLAB-doc-style mixed problem: x2 integer
[x, f, ef, out] = intlinprog([-3; -2; -1], 3, [1 1 1], 7, [4 2 1], 12, zeros(3, 1), [inf; inf; 1], [], o);
rep('M2_mixed_eq', x, f, ef, out, 0);
% M3 assignment 5x5 (binary, equalities), cost c(i,j)
C = mod((1:5)' * (1:5) * 7 + (1:5)', 11) + 1;
Aeq = [kron(eye(5), ones(1, 5)); kron(ones(1, 5), eye(5))];
tic; [x, f, ef, out] = intlinprog(C(:), 1:25, [], [], Aeq, ones(10, 1), zeros(25, 1), ones(25, 1), [], o); t = toc;
rep('M3_assignment5', find(round(x)).', f, ef, out, t);
% M4 set covering: 15 elements, 20 sets
S = mod((1:15)' * (1:20) + (1:15)' .^ 2, 4) == 0 | eye(15, 20) == 1;
cost = mod((1:20)', 5) + 1;
tic; [x, f, ef, out] = intlinprog(cost, 1:20, -double(S), -ones(15, 1), [], [], zeros(20, 1), ones(20, 1), [], o); t = toc;
rep('M4_setcover', find(round(x)).', f, ef, out, t);
% M5 general integers with negative bounds
[x, f, ef, out] = intlinprog([1; -2; 3; -1], [1 2 4], [1 1 1 1; -1 2 -1 1; 3 -1 0 2], [10; 4; 12], [], [], [-5; -5; 0; -5], [5; 5; 2.5; 5], [], o);
rep('M5_general_int', x, f, ef, out, 0);
% M6 infeasible integer problem (LP feasible): 2*x1 + 2*x2 = 3, x integer
[x, f, ef] = intlinprog([1; 1], 1:2, [], [], [2 2], 3, [0; 0], [5; 5], [], o);
fprintf('MILP M6_int_infeasible ef=%d\n', ef);
% M7 unbounded relaxation
[x, f, ef] = intlinprog([-1; 0], 1, [1 -1], 1, [], [], [0; 0], [], [], o);
fprintf('MILP M7_unbounded ef=%d\n', ef);
% M8 x0 as incumbent + MaxNodes stop
try
    o8 = optimoptions(o, 'MaxNodes', 1, 'CutGeneration', 'none');
catch
    o8 = optimoptions(o, 'MaxNodes', 1);     % releases whose intlinprog has no CutGeneration option
end
[x, f, ef, out] = intlinprog(-v, 1:12, w, 165, [], [], zeros(12, 1), ones(12, 1), zeros(12, 1), o8);
fprintf('MILP M8_maxnodes ef=%d f=%g nodes=%d gap=%.3g%%\n', ef, f, out.numnodes, out.relativegap);
% M9 problem struct
pr = struct('f', [-1; -1], 'intcon', [1 2], 'Aineq', [2 1; 1 2], 'bineq', [4.5; 4.5], 'lb', [0; 0], 'ub', [], 'options', o, 'solver', 'intlinprog');
[x, f, ef, out] = intlinprog(pr);
rep('M9_struct', x, f, ef, out, 0);
% M10 strongly correlated knapsack, 30 items (hard for branch and bound)
w = mod((1:30) * 37, 61) + 20; v = w + 10; cap = floor(sum(w) / 2);
tic; [x, f, ef, out] = intlinprog(-v(:), 1:30, w, cap, [], [], zeros(30, 1), ones(30, 1), [], o); t = toc;
fprintf('MILP M10_knapsack30 ef=%d f=%.8g weight=%d/%d | nodes=%d time=%.3fs\n', ef, f, round(w * x), cap, out.numnodes, t);
% M11 general-integer MILP, 15 variables in [0, 10], 10 constraints. Weak LP relaxation: it needs
% ~22000 nodes (12 s) to prove optimality, so it is stopped at 2000 nodes here; the incumbent must
% already be the optimum (f* = -206, from HiGHS).
m = 10; n = 15;
Am = mod((1:m)' * (1:n) * 13 + (1:m)', 17) - 5;
bm = 20 + mod((1:m)' * 7, 23);
cm = -(mod((1:n)' * 11, 9) + 1);
tic; [x, f, ef, out] = intlinprog(cm, 1:n, Am, bm, [], [], zeros(n, 1), 10 * ones(n, 1), [], optimoptions(o, 'MaxNodes', 2000)); t = toc;
fprintf('MILP M11_general15 ef=%d f=%.8g gap=%.2f%% | nodes=%d time=%.3fs viol=%.1e\n', ef, f, out.relativegap, out.numnodes, t, out.constrviolation);
% M12 general-integer MILP, 8 variables in [0, 6], 6 constraints (closes the gap)
m = 6; n = 8;
Am = mod((1:m)' * (1:n) * 5 + (1:m)', 11) - 3; bm = 10 + mod((1:m)' * 7, 13); cm = -(mod((1:n)' * 7, 5) + 1);
tic; [x, f, ef, out] = intlinprog(cm, 1:n, Am, bm, [], [], zeros(n, 1), 6 * ones(n, 1), [], o); t = toc;
fprintf('MILP M12_general8 ef=%d f=%.8g x=%s | nodes=%d time=%.3fs\n', ef, f, mat2str(round(x(:).')), out.numnodes, t);
% M13 Gomory cuts: general-integer MILP where root cuts cut the tree from 5633 to 1831 nodes
m = 9; n = 11; s = 3;
Am = mod((1:m)' * (1:n) * (s + 6) + (1:m)' * s, 13) - 4; bm = 15 + mod((1:m)' * (s + 3), 19); cm = -(mod((1:n)' * (s + 4), 7) + 1);
try
    on = optimoptions(o, 'CutGeneration', 'none'); ob = optimoptions(o, 'CutGeneration', 'basic');
catch
    on = o; ob = o;                           % no CutGeneration option in this release
end
[x, f1, ef1, o1] = intlinprog(cm, 1:n, Am, bm, [], [], zeros(n, 1), 8 * ones(n, 1), [], on);
[x, f2, ef2, o2] = intlinprog(cm, 1:n, Am, bm, [], [], zeros(n, 1), 8 * ones(n, 1), [], ob);
if ~isfield(o2, 'numcuts'), o2.numcuts = NaN; end
fprintf('MILP M13_cuts f=%g/%g ef=%d/%d nodes none=%d basic=%d cuts=%d (f* = -30)\n', f1, f2, ef1, ef2, o1.numnodes, o2.numnodes, o2.numcuts);

function rep(name, x, f, ef, out, t)
fprintf('MILP %s ef=%d f=%.8g x=%s | nodes=%d time=%.3fs\n', name, ef, f, mat2str(round(x(:).' * 1e6) / 1e6), out.numnodes, t);
end
