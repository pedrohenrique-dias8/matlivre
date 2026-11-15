function [x, status, info] = optim_simplex(c, A, b, Aeq, beq, lb, ub, opts)
%OPTIM_SIMPLEX  Dense bounded dual simplex with warm start (MatLivre optimization toolbox).
%
%   [x, status, info] = optim_simplex(c, A, b, Aeq, beq, lb, ub)
%   [x, status, info] = optim_simplex(..., opts)
%
%   minimize c'*x  subject to  A*x <= b,  Aeq*x = beq,  lb <= x <= ub.
%
%   The problem is written as [A; Aeq]*x + s = [b; beq] with slacks s in [0, Inf) for the
%   inequalities and s = 0 for the equalities. Infinite bounds are replaced by an artificial box
%   (+-BIG). With every variable boxed, any basis is made dual feasible by moving each nonbasic
%   variable to the bound that matches the sign of its reduced cost, so the dual simplex needs no
%   phase 1. Leaving row: largest primal infeasibility. Entering column: dual ratio test, ties
%   broken by the largest pivot (and by the smallest index after a run of degenerate pivots).
%
%   status: 1 optimal, -2 infeasible, -3 unbounded, 0 iteration limit.
%   info:   .basis (warm start: struct with B, the basic indices into [x; s], and stat, -1 at
%           lower / +1 at upper / 0 basic, for every column), .y (row duals: c_B' * inv(B)),
%           .d (reduced costs of [x; s]), .s (slacks), .iterations, .objective, .atart (logical:
%           column sits at an artificial bound).
%   opts:   .basis (warm start), .maxit (default 50*(m+n) + 1000), .tol (1e-9 primal, scaled).
%
%   Reference: A. Koberstein (2005), "The dual simplex method, techniques for a fast and stable
%   implementation" (PhD thesis, Paderborn); Nocedal and Wright, Ch. 13. Clean-room.
if nargin < 8, opts = struct(); end
c = double(c(:)); n = numel(c);
if isempty(A), A = zeros(0, n); b = zeros(0, 1); end
if isempty(Aeq), Aeq = zeros(0, n); beq = zeros(0, 1); end
mA = size(A, 1); mE = size(Aeq, 1); m = mA + mE; N = n + m;
M = [A; Aeq]; rhs = [b(:); beq(:)];
Mf = [M, eye(m)];
cz = [c; zeros(m, 1)];
lz = [lb(:); zeros(m, 1)];
uz = [ub(:); inf(mA, 1); zeros(mE, 1)];
dat = [abs(M(:)); abs(rhs); abs(c); abs(lz(isfinite(lz))); abs(uz(isfinite(uz)))];
BIG = 1e7 * max([1; dat]);
la = lz; la(~isfinite(la)) = -BIG;
ua = uz; ua(~isfinite(ua)) = BIG;
art_l = ~isfinite(lz); art_u = ~isfinite(uz);
tolp = sp_get(opts, 'tol', 1e-9) * max(1, norm(rhs, inf));
told = 1e-9 * max(1, norm(c, inf));
tolpiv = 1e-9;
maxit = sp_get(opts, 'maxit', 50 * (m + n) + 1000);

% ---- initial basis: warm start or all slacks
stat = zeros(N, 1);
ws = sp_get(opts, 'basis', []);
if ~isempty(ws) && numel(ws.stat) == N && numel(ws.B) == m
    B = ws.B(:); stat = ws.stat(:);
else
    B = (n + 1:N).';
    stat(1:n) = -1;
    stat(B) = 0;
end
isb = false(N, 1); isb(B) = true;
it = 0; status = 0; degen = 0; lastobj = -inf;
while it < maxit
    it = it + 1;
    MB = Mf(:, B);
    [L, U, P] = lu(MB);
    y = (P.' * (L.' \ (U.' \ cz(B))));          % y = MB' \ c_B
    d = cz - Mf.' * y; d(B) = 0;
    % dual feasibility by bound flipping (every nonbasic variable is boxed)
    nb = find(~isb);
    flipl = nb(stat(nb) == 1 & d(nb) > told);
    flipu = nb(stat(nb) == -1 & d(nb) < -told);
    stat(flipl) = -1; stat(flipu) = 1;
    fixed = nb(la(nb) == ua(nb)); stat(fixed(stat(fixed) == 0)) = -1;
    xz = zeros(N, 1);
    xz(stat == -1) = la(stat == -1); xz(stat == 1) = ua(stat == 1);
    xB = U \ (L \ (P * (rhs - Mf(:, ~isb) * xz(~isb))));
    xz(B) = xB;
    % ---- leaving row: largest primal infeasibility
    lo = la(B) - xB; hi = xB - ua(B);
    viol = max(lo, hi);
    [vmax, r] = max(viol);
    if isempty(vmax) || vmax <= tolp
        status = 1; break
    end
    below = lo(r) > hi(r);
    er = zeros(m, 1); er(r) = 1;
    rho = P.' * (L.' \ (U.' \ er));               % row r of inv(MB)
    alpha = (rho.' * Mf).';                       % row r of inv(MB)*Mf
    cand = ~isb & la ~= ua;
    if below
        cand = cand & ((stat == -1 & alpha < -tolpiv) | (stat == 1 & alpha > tolpiv));
    else
        cand = cand & ((stat == -1 & alpha > tolpiv) | (stat == 1 & alpha < -tolpiv));
    end
    J = find(cand);
    if isempty(J)
        status = -2; break                        % dual unbounded: primal infeasible
    end
    ratio = abs(d(J)) ./ abs(alpha(J));
    rmin = min(ratio);
    tie = J(ratio <= rmin + 1e-12 * max(1, rmin));
    if degen > 50
        q = min(tie);                             % anti-cycling fallback
    else
        [~, k] = max(abs(alpha(tie))); q = tie(k);
    end
    obj = cz.' * xz;
    if abs(obj - lastobj) <= 1e-14 * max(1, abs(obj)), degen = degen + 1; else, degen = 0; end
    lastobj = obj;
    % ---- pivot
    leave = B(r);
    if below, stat(leave) = -1; else, stat(leave) = 1; end
    isb(leave) = false; isb(q) = true;
    B(r) = q; stat(q) = 0;
end
% final values with the last basis
MB = Mf(:, B); [L, U, P] = lu(MB);
y = (P.' * (L.' \ (U.' \ cz(B)))); d = cz - Mf.' * y; d(B) = 0;
xz = zeros(N, 1); xz(stat == -1) = la(stat == -1); xz(stat == 1) = ua(stat == 1);
xz(B) = U \ (L \ (P * (rhs - Mf(:, ~isb) * xz(~isb))));
atart = (stat == -1 & art_l) | (stat == 1 & art_u) | ...
    (isb & (abs(xz) >= 0.5 * BIG));
if status == 1 && any(atart & (abs(d) > told | isb))
    status = -3;                                  % optimum of the boxed LP on the artificial box
end
x = xz(1:n);
info = struct('basis', struct('B', B, 'stat', stat), 'y', y, 'd', d, 's', xz(n + 1:end), ...
    'iterations', it, 'objective', c.' * x, 'atart', atart, 'BIG', BIG, 'xz', xz, ...
    'la', la, 'ua', ua);
end

function v = sp_get(s, name, default)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end
