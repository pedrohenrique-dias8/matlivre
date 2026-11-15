function bench_many(rule)
% 36 MILPs (general integers, multidimensional knapsacks, facility location); prints the shifted
% geometric means (shift 10 nodes, 0.01 s) of nodes and time, and the number solved.
o = optimoptions('intlinprog', 'Display', 'off', 'BranchRule', rule, 'MaxNodes', 60000);
N = []; T = []; solved = 0;
for k = 1:14                                   % general integers
    m = 6 + mod(k, 5); n = 9 + mod(3 * k, 8);
    c = -(mod((1:n)' * (k + 5), 7 + mod(k, 3)) + 1);
    A = mod((1:m)' * (1:n) * (k + 9) + (1:m)' * (k + 2), 13 + mod(k, 4)) - 4;
    b = 15 + mod((1:m)' * (k + 4), 19);
    [N, T, solved] = run1(c, A, b, [], [], 1:n, zeros(n, 1), (6 + mod(k, 5)) * ones(n, 1), o, N, T, solved);
end
for k = 1:12                                   % 0/1 multidimensional knapsacks
    m = 2 + mod(k, 4); n = 20 + 2 * k;
    W = mod((1:m)' * (1:n) * (k + 2) + (1:n) * (5 + mod(k, 3)), 37 + mod(k, 5)) + 8;
    c = -(mod((1:n)' * (k + 10), 31) + 15);
    [N, T, solved] = run1(c, W, floor(sum(W, 2) / (2 + mod(k, 2))), [], [], 1:n, zeros(n, 1), ones(n, 1), o, N, T, solved);
end
for k = 1:10                                   % capacitated facility location, big-M
    nf = 4 + mod(k, 4); nc = 8 + k;
    d = mod((1:nc)' * (k + 6), 9) + 3; fc = 30 + mod((1:nf)' * (k + 12), 29);
    tc = mod((1:nf)' * (1:nc) * (k + 4) + (1:nf)', 15) + 1; n = nf + nf * nc;
    Aeq = [zeros(nc, nf), kron(eye(nc), ones(1, nf))];
    A = [-1e4 * eye(nf), kron(ones(1, nc), eye(nf)); zeros(nf, nf), kron(ones(1, nc), eye(nf))];
    b = [zeros(nf, 1); (0.35 + 0.05 * mod(k, 3)) * sum(d) * ones(nf, 1)];
    [N, T, solved] = run1([fc; tc(:)], A, b, Aeq, d, 1:nf, zeros(n, 1), [ones(nf, 1); inf(nf * nc, 1)], o, N, T, solved);
end
sgm = @(v, s) exp(mean(log(v + s))) - s;
fprintf('%-14s solved=%d/%d  nodes sgm=%.1f total=%d  time sgm=%.4f total=%.2f\n', rule, solved, numel(N), sgm(N, 10), sum(N), sgm(T, 0.01), sum(T));
end

function [N, T, solved] = run1(c, A, b, Aeq, beq, ic, lb, ub, o, N, T, solved)
t0 = tic; [~, ~, ef, out] = intlinprog(c, ic, A, b, Aeq, beq, lb, ub, [], o); t = toc(t0);
N(end + 1) = out.numnodes; T(end + 1) = t; solved = solved + (ef == 1);
end
