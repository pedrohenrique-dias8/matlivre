function bench_hard(rule)
% 12 harder general-integer MILPs (15-20 variables in [0, 8..10], 9-13 rows).
o = optimoptions('intlinprog', 'Display', 'off', 'BranchRule', rule, 'MaxNodes', 100000);
N = []; T = []; solved = 0;
for k = 1:12
    m = 9 + mod(k, 5); n = 15 + mod(2 * k, 6);
    c = -(mod((1:n)' * (k + 6), 8 + mod(k, 3)) + 1);
    A = mod((1:m)' * (1:n) * (k + 10) + (1:m)' * k, 15 + mod(k, 3)) - 4;
    b = 18 + mod((1:m)' * (k + 5), 21);
    t0 = tic; [~, ~, ef, out] = intlinprog(c, 1:n, A, b, [], [], zeros(n, 1), (8 + mod(k, 3)) * ones(n, 1), [], o); t = toc(t0);
    N(end + 1) = out.numnodes; T(end + 1) = t; solved = solved + (ef == 1);
end
sgm = @(v, s) exp(mean(log(v + s))) - s;
fprintf('%-14s solved=%d/%d  nodes sgm=%.0f total=%d  time sgm=%.3f total=%.2f\n', rule, solved, numel(N), sgm(N, 10), sum(N), sgm(T, 0.01), sum(T));
end
