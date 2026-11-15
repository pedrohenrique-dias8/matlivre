% Node counts and times for the stage-A comparison (deterministic problems).
o = optimoptions('intlinprog', 'Display', 'off');
P = {};
% M11: 15 general integers in [0,10], 10 rows
m = 10; n = 15; P{end+1} = {'M11', -(mod((1:n)' * 11, 9) + 1), mod((1:m)' * (1:n) * 13 + (1:m)', 17) - 5, 20 + mod((1:m)' * 7, 23), 10};
% M13: 11 general integers in [0,8], 9 rows
m = 9; n = 11; s = 3; P{end+1} = {'M13', -(mod((1:n)' * (s + 4), 7) + 1), mod((1:m)' * (1:n) * (s + 6) + (1:m)' * s, 13) - 4, 15 + mod((1:m)' * (s + 3), 19), 8};
% G1..G4: more general-integer MILPs of the same family
for k = 1:4
    m = 8 + k; n = 10 + 2 * k;
    P{end+1} = {sprintf('G%d', k), -(mod((1:n)' * (k + 6), 8) + 1), mod((1:m)' * (1:n) * (k + 10) + (1:m)' * k, 15) - 4, 18 + mod((1:m)' * (k + 5), 21), 9};
end
% K1..K3: multidimensional 0/1 knapsacks
for k = 1:3
    m = 3 + k; n = 25 + 5 * k;
    W = mod((1:m)' * (1:n) * (k + 3) + (1:n) * 7, 41) + 10;
    P{end+1} = {sprintf('K%d', k), -(mod((1:n)' * (k + 11), 37) + 20), W, floor(sum(W, 2) / 3), 1};
end
tot = 0; totn = 0;
for i = 1:numel(P)
    p = P{i}; n = numel(p{2});
    t0 = tic; [x, f, ef, out] = intlinprog(p{2}, 1:n, p{3}, p{4}, [], [], zeros(n, 1), p{5} * ones(n, 1), [], o); t = toc(t0);
    fprintf('%-4s ef=%d f=%g nodes=%d time=%.3fs\n', p{1}, ef, f, out.numnodes, t);
    tot = tot + t; totn = totn + out.numnodes;
end
fprintf('TOTAL nodes=%d time=%.3fs\n', totn, tot);
