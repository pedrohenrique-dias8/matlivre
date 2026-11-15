% Capacitated facility location with big-M capacity rows (structure that presolve can use).
% y_i binary (open), x_ij >= 0 continuous (shipped), sum_i x_ij = d_j, sum_j x_ij <= M*y_i,
% plus a few singleton rows and a fixed facility.
function bench_fl(pre)
o = optimoptions('intlinprog', 'Display', 'off', 'IntegerPreprocess', pre);
tot = 0; totn = 0;
for k = 1:3
    nf = 5 + 2 * k; nc = 10 + 4 * k;
    d = mod((1:nc)' * (k + 7), 9) + 3;
    fc = 40 + mod((1:nf)' * (k + 13), 31);
    cap = 0.45 * sum(d) * ones(nf, 1);
    tc = mod((1:nf)' * (1:nc) * (k + 5) + (1:nf)', 17) + 1;
    M = 1e4;
    n = nf + nf * nc;                          % [y; x(:)] with x nf-by-nc, column-major
    f = [fc; tc(:)];
    Aeq = [zeros(nc, nf), kron(eye(nc), ones(1, nf))]; beq = d;
    A = [-M * eye(nf), kron(ones(1, nc), eye(nf))]; b = zeros(nf, 1);
    A = [A; zeros(nf, nf), kron(ones(1, nc), eye(nf))]; b = [b; cap];   % capacities
    A = [A; [1, zeros(1, n - 1)]; [zeros(1, nf), 1, zeros(1, n - nf - 1)]]; b = [b; 1; 5];   % singletons
    lb = zeros(n, 1); ub = [ones(nf, 1); inf(nf * nc, 1)]; lb(nf) = 1;   % last facility forced open
    t0 = tic; [x, fv, ef, out] = intlinprog(f, 1:nf, A, b, Aeq, beq, lb, ub, [], o); t = toc(t0);
    fprintf('F%d  ef=%d f=%g nodes=%d time=%.3fs\n', k, ef, fv, out.numnodes, t);
    tot = tot + t; totn = totn + out.numnodes;
end
fprintf('TOTAL nodes=%d time=%.3fs\n', totn, tot);
end
