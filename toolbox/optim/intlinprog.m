function [x, fval, exitflag, output] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, x0, options)
%INTLINPROG  Mixed-integer linear programming.
%   x = intlinprog(f, intcon, A, b)             minimizes f'*x, x(intcon) integer, A*x <= b
%   x = intlinprog(f, intcon, A, b, Aeq, beq)   and Aeq*x = beq
%   x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub)
%   x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, x0)
%   x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, x0, options)
%   x = intlinprog(problem)
%   [x, fval, exitflag, output] = intlinprog(...)
%
%   LP-based branch and bound. Relaxations are solved by the bounded dual simplex
%   (optim_simplex), each node warm-started from its parent's optimal basis. At the root, rounds
%   of Gomory mixed-integer cuts are added (CutGeneration 'basic' = 5 rounds, 'intermediate'
%   = 10, 'advanced' = 20, 'none'; at most 20 cuts per round). Search: depth-first dives until an integer-feasible point is found, then best-bound
%   node selection (NodeSelection 'minobj'), or depth-first throughout ('mininfeas'). Branching
%   variable: BranchRule 'maxfun' (default here; MATLAB's default is 'reliability'), 'reliability'
%   (pseudocost product rule, with strong branching on up to 8 candidates whose pseudocosts
%   have fewer than 2 observations), 'maxpscost' (pseudocosts only, initialised from |f(j)|),
%   'maxfun' (largest |f(j)| among fractional variables) or 'mostfractional'. No presolve, and cuts only
%   at the root: problems with weak LP relaxations can need many more nodes than in MATLAB. When the objective
%   can only take integer values, nodes whose bound rounds up to the incumbent are pruned. At every node a rounding heuristic tries the rounded LP
%   point. x0, if integer feasible, is used as the first incumbent.
%
%   Options (optimoptions('intlinprog', ...)): MaxNodes (1e7), MaxTime (7200),
%   RelativeGapTolerance (1e-4), AbsoluteGapTolerance (0), IntegerTolerance (1e-5),
%   ConstraintTolerance (1e-4), MaxFeasiblePoints (Inf), BranchRule, NodeSelection,
%   Display ('iter' | 'final' | 'off'), OutputFcn (stop = f(x, optimValues, state), with
%   optimValues.phase 'branching' and the fields fval, numnodes, numfeaspoints, relativegap).
%   exitflag:  1 optimal solution (gap within tolerance), 2 stopped with an integer feasible
%              point (limits or output function), 0 stopped with no integer feasible point,
%             -1 stopped by an output function before a feasible point, -2 infeasible,
%             -3 LP relaxation unbounded.
%   output: relativegap (percent: 100*(U - L)/(abs(U) + 1)), absolutegap, numfeaspoints,
%   numnodes, numcuts, constrviolation, message.
%
%   Reference: A. H. Land and A. G. Doig (1960); L. A. Wolsey, Integer Programming (1998), Ch. 7.
%   Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(f)
    p = f;
    f = ilp_fld(p, 'f'); intcon = ilp_fld(p, 'intcon'); A = ilp_fld(p, 'Aineq'); b = ilp_fld(p, 'bineq');
    Aeq = ilp_fld(p, 'Aeq'); beq = ilp_fld(p, 'beq'); lb = ilp_fld(p, 'lb'); ub = ilp_fld(p, 'ub');
    x0 = ilp_fld(p, 'x0'); options = ilp_fld(p, 'options'); nin = 10;
else
    nin = nargin;
end
if nin < 3, A = []; end
if nin < 4, b = []; end
if nin < 5, Aeq = []; end
if nin < 6, beq = []; end
if nin < 7, lb = []; end
if nin < 8, ub = []; end
if nin < 9, x0 = []; end
if nin < 10 || isempty(options), options = struct(); end
f = double(f(:)); n = numel(f);
intcon = unique(intcon(:)).';
if isempty(A), A = zeros(0, n); b = zeros(0, 1); end
if isempty(Aeq), Aeq = zeros(0, n); beq = zeros(0, 1); end
b = double(b(:)); beq = double(beq(:));
if isempty(lb), lb = -inf(n, 1); end
if isempty(ub), ub = inf(n, 1); end
lb = double(lb(:)); ub = double(ub(:));
lb(intcon) = ceil(lb(intcon) - 1e-9); ub(intcon) = floor(ub(intcon) + 1e-9);

maxnodes = optim_getopt(options, {'MaxNodes'}, 1e7);
maxtime = optim_getopt(options, {'MaxTime'}, 7200);
rgap = optim_getopt(options, {'RelativeGapTolerance', 'TolGapRel'}, 1e-4);
agap = optim_getopt(options, {'AbsoluteGapTolerance', 'TolGapAbs'}, 0);
itol = optim_getopt(options, {'IntegerTolerance', 'TolInteger'}, 1e-5);
ctol = optim_getopt(options, {'ConstraintTolerance', 'TolCon'}, 1e-4);
maxfeas = optim_getopt(options, {'MaxFeasiblePoints'}, inf);
brule = lower(char(optim_getopt(options, {'BranchRule', 'BranchingRule'}, 'maxfun')));
nsel = lower(char(optim_getopt(options, {'NodeSelection'}, 'minobj')));
lvl = optim_display(options, 'iter');
OF = optim_outfcn('setup', options);
% if the objective can only take integer values, a node whose bound rounds up to the incumbent
% cannot improve it
objint = all(f(setdiff(1:n, intcon)) == 0) && all(abs(f(intcon) - round(f(intcon))) < 1e-12);
t0 = tic;

output = struct('relativegap', [], 'absolutegap', [], 'numfeaspoints', 0, 'numnodes', 0, ...
    'constrviolation', [], 'message', '', 'numcuts', 0);
U = inf; xbest = []; nfeas = 0;
% incumbent from x0
if ~isempty(x0)
    xs = double(x0(:));
    if numel(xs) == n && ilp_feasible(xs, A, b, Aeq, beq, lb, ub, intcon, itol, ctol)
        xs(intcon) = round(xs(intcon));
        U = f.' * xs; xbest = xs; nfeas = 1;
    end
end

% root relaxation (dual simplex)
Ao = A; bo = b;                                   % original rows, for feasibility checks
if any(lb > ub)
    [x, fval, exitflag, output] = ilp_finish([], [], -2, output, 'No feasible solution found: lb > ub after rounding the integer bounds.', lvl);
    return
end
[xr, ef, rinfo] = optim_simplex(f, A, b, Aeq, beq, lb, ub);
if ef == -2
    [x, fval, exitflag, output] = ilp_finish([], [], -2, output, 'No feasible solution found: the LP relaxation is infeasible.', lvl);
    return
end
if ef == -3
    [x, fval, exitflag, output] = ilp_finish([], [], -3, output, 'Problem is unbounded: the LP relaxation is unbounded.', lvl);
    return
end
rootlp = f.' * xr;
% ---- Gomory mixed-integer cuts at the root
ncuts = 0;
cutmode = lower(char(optim_getopt(options, {'CutGeneration'}, 'basic')));
rounds = 0;
switch cutmode
    case 'none', rounds = 0;
    case 'basic', rounds = 5;
    case 'intermediate', rounds = 10;
    otherwise, rounds = 20;   % 'advanced'
end
stallr = 0;
for rd = 1:rounds
    [Ac, bc] = ilp_gmi(rinfo, A, b, Aeq, beq, intcon, n, 20);
    if isempty(Ac), break, end
    k = size(Ac, 1); mA = size(A, 1);
    A = [A; Ac]; b = [b; bc]; ncuts = ncuts + k;
    % warm start: the new slacks are basic, the equality slacks move by k positions
    Bs = rinfo.basis.B; st = rinfo.basis.stat;
    Bs(Bs > n + mA) = Bs(Bs > n + mA) + k;
    st = [st(1:n + mA); zeros(k, 1); st(n + mA + 1:end)];
    ws = struct('B', [Bs; (n + mA + (1:k)).'], 'stat', st);
    fold = f.' * xr;
    [xr2, ef2, rinfo2] = optim_simplex(f, A, b, Aeq, beq, lb, ub, struct('basis', ws));
    if ef2 ~= 1, A = A(1:mA, :); b = b(1:mA); ncuts = ncuts - k; break, end   % numerical trouble: drop the round
    xr = xr2; rinfo = rinfo2;
    if f.' * xr - fold <= 1e-6 * max(1, abs(fold))
        stallr = stallr + 1;                      % degenerate LPs can keep the value for a while
        if stallr >= 3, break, end
    else
        stallr = 0;
    end
end
rootbasis = rinfo.basis;
if lvl == 2
    fprintf('\nLP relaxation: f = %.6g; after %d Gomory cuts: f = %.6g\n\n', rootlp, ncuts, f.' * xr);
    fprintf('    Nodes      Total time (s)   Num int solution   Integer fval     Relative gap (%%)\n');
end
% node list, stored column-wise in preallocated arrays (struct arrays are slow to grow):
% bounds, LP bound of the parent, depth, and the branching that created the node
cap = 1024;
NL = zeros(n, cap); NU = zeros(n, cap); NB = zeros(1, cap); ND = zeros(1, cap);
NV = zeros(1, cap); NR = zeros(1, cap); NS = zeros(1, cap);
mrow = size(A, 1) + size(Aeq, 1); Ncol = n + mrow;
NBB = zeros(mrow, cap); NBS = zeros(Ncol, cap);    % parent basis of each node (warm start)
NL(:, 1) = lb; NU(:, 1) = ub; NB(1) = f.' * xr; cnt = 1;
NBB(:, 1) = rootbasis.B; NBS(:, 1) = rootbasis.stat;
% pseudocosts: average objective increase per unit change, for down (1) and up (2) branches
pc = zeros(n, 2); pcn = zeros(n, 2);
nnodes = 0; stopreason = ''; userstop = false;
if OF.active
    [s1, OF] = optim_outfcn(OF, [], ilp_ov(U, nnodes, nfeas, inf, 'init'), 'init');
    userstop = s1;
end
L = NB(1);
while cnt > 0 && ~userstop
    % ---- pick a node: depth-first until the first incumbent (or always, for 'mininfeas'),
    %      best bound afterwards (ties to the deeper node)
    if isinf(U) || strcmp(nsel, 'mininfeas') || strcmp(nsel, 'depthfirst')
        [~, k] = max(ND(1:cnt) - 1e-9 * NB(1:cnt));
    else
        [~, k] = min(NB(1:cnt) - 1e-9 * ND(1:cnt));
    end
    nlb = NL(:, k); nub = NU(:, k); nbound = NB(k); ndepth = ND(k);
    bvar = NV(k); bdir = NR(k); bdist = NS(k);
    pbasis = struct('B', NBB(:, k), 'stat', NBS(:, k));
    NL(:, k) = NL(:, cnt); NU(:, k) = NU(:, cnt); NB(k) = NB(cnt); ND(k) = ND(cnt);
    NV(k) = NV(cnt); NR(k) = NR(cnt); NS(k) = NS(cnt);
    NBB(:, k) = NBB(:, cnt); NBS(:, k) = NBS(:, cnt);
    cnt = cnt - 1;
    if ilp_prune(nbound, U, agap, rgap, objint), continue, end   % pruned by bound
    % ---- solve the relaxation of the node (the root was solved already)
    if nnodes == 0
        xn = xr; efn = 1; nbasis = rootbasis;
    else
        [xn, efn, nbasis] = ilp_lp(f, A, b, Aeq, beq, nlb, nub, pbasis);
    end
    nnodes = nnodes + 1;
    if efn > 0
        fn = f.' * xn;
        if bvar > 0 && bdist > 0                                       % update pseudocosts
            pc(bvar, bdir) = pc(bvar, bdir) + max(fn - nbound, 0) / bdist;
            pcn(bvar, bdir) = pcn(bvar, bdir) + 1;
        end
    end
    if efn > 0 && ~ilp_prune(fn, U, agap, rgap, objint)
        % ---- rounding heuristic
        xh = xn; xh(intcon) = round(xh(intcon));
        if f.' * xh < U && ilp_feasible(xh, Ao, bo, Aeq, beq, lb, ub, intcon, itol, ctol)
            U = f.' * xh; xbest = xh; nfeas = nfeas + 1;
        end
        % ---- integrality of the node solution
        frac = abs(xn(intcon) - round(xn(intcon)));
        isf = find(frac > itol);
        if isempty(isf)
            xi = xn; xi(intcon) = round(xi(intcon));
            if f.' * xi < U
                U = f.' * xi; xbest = xi; nfeas = nfeas + 1;
            end
        else
            % ---- branching variable
            cand = intcon(isf);
            fl = xn(cand) - floor(xn(cand));          % distance to the down branch
            switch brule
                case 'mostfractional'
                    [~, j] = max(fl .* (1 - fl));
                case 'maxfun'
                    [~, j] = max(abs(f(cand)) .* (1 + frac(isf)));
                otherwise   % 'reliability' / 'maxpscost': pseudocost product score
                    if strcmp(brule, 'reliability')
                        % strong branching on candidates whose pseudocosts are not yet reliable
                        % (fewer than 2 observations per direction), at most 8 per node
                        unrel = find(min(pcn(cand, :), [], 2) < 2);
                        if numel(unrel) > 8
                            [~, o8] = sort(abs(fl(unrel) - 0.5)); unrel = unrel(o8(1:8));
                        end
                        for q = unrel(:).'
                            vq = cand(q);
                            u1 = nub; u1(vq) = floor(xn(vq));
                            [xd, okd] = ilp_lp(f, A, b, Aeq, beq, nlb, u1, nbasis);
                            l2 = nlb; l2(vq) = ceil(xn(vq));
                            [xu, oku] = ilp_lp(f, A, b, Aeq, beq, l2, nub, nbasis);
                            nnodes = nnodes + 0;   % strong-branching LPs are not counted as nodes
                            if okd, gd = max(f.' * xd - fn, 0); else, gd = 1e6 * (1 + abs(fn)); end
                            if oku, gu = max(f.' * xu - fn, 0); else, gu = 1e6 * (1 + abs(fn)); end
                            pc(vq, 1) = pc(vq, 1) + gd / max(fl(q), 1e-9); pcn(vq, 1) = pcn(vq, 1) + 1;
                            pc(vq, 2) = pc(vq, 2) + gu / max(1 - fl(q), 1e-9); pcn(vq, 2) = pcn(vq, 2) + 1;
                        end
                    end
                    avg = mean(abs(f(intcon))) + eps;
                    qd = pc(cand, 1) ./ max(pcn(cand, 1), 1); qd(pcn(cand, 1) == 0) = abs(f(cand(pcn(cand, 1) == 0))) + avg;
                    qu = pc(cand, 2) ./ max(pcn(cand, 2), 1); qu(pcn(cand, 2) == 0) = abs(f(cand(pcn(cand, 2) == 0))) + avg;
                    [~, j] = max(max(qd .* fl, 1e-6) .* max(qu .* (1 - fl), 1e-6));
            end
            v = cand(j); xv = xn(v);
            if cnt + 2 > size(NL, 2)                  % grow the storage
                NL = [NL, zeros(n, cap)]; NU = [NU, zeros(n, cap)]; NB = [NB, zeros(1, cap)];
                ND = [ND, zeros(1, cap)]; NV = [NV, zeros(1, cap)]; NR = [NR, zeros(1, cap)];
                NS = [NS, zeros(1, cap)]; NBB = [NBB, zeros(mrow, cap)]; NBS = [NBS, zeros(Ncol, cap)];
                cap = 2 * cap;
            end
            % down branch, then up branch (the side closer to the LP value is pushed last,
            % so depth-first search takes it first)
            lo_ub = nub; lo_ub(v) = floor(xv); hi_lb = nlb; hi_lb(v) = ceil(xv);
            order = [1 2]; if xv - floor(xv) >= 0.5, order = [2 1]; end
            for dir = order
                cnt = cnt + 1;
                if dir == 1
                    NL(:, cnt) = nlb; NU(:, cnt) = lo_ub; NS(cnt) = xv - floor(xv);
                else
                    NL(:, cnt) = hi_lb; NU(:, cnt) = nub; NS(cnt) = ceil(xv) - xv;
                end
                NB(cnt) = fn; ND(cnt) = ndepth + 1; NV(cnt) = v; NR(cnt) = dir;
                NBB(:, cnt) = nbasis.B; NBS(:, cnt) = nbasis.stat;
            end
        end
    end
    if cnt > 0, L = min(NB(1:cnt)); else, L = U; end
    if lvl == 2 && (mod(nnodes, 500) == 0 || nnodes <= 10)
        fprintf('%9d %16.2f %18d %16.6g %18.3g\n', nnodes, toc(t0), nfeas, U, ilp_gap(U, L));
    end
    if isfinite(U) && (U - L <= agap || (U - L) <= rgap * (abs(U) + 1) || (objint && ceil(L - 1e-6) >= U)), cnt = 0; break, end
    if OF.active
        [userstop, OF] = optim_outfcn(OF, xbest, ilp_ov(U, nnodes, nfeas, ilp_gap(U, L), 'branching'), 'iter');
        if userstop, stopreason = 'output function'; break, end
    end
    if nfeas >= maxfeas, stopreason = 'MaxFeasiblePoints'; break, end
    if nnodes >= maxnodes, stopreason = 'MaxNodes'; break, end
    if toc(t0) > maxtime, stopreason = 'MaxTime'; break, end
end
if cnt > 0, L = min([NB(1:cnt), U]); else, L = U; end
output.numnodes = nnodes; output.numfeaspoints = nfeas; output.numcuts = ncuts;
if isempty(xbest)
    if strcmp(stopreason, 'output function')
        [x, fval, exitflag, output] = ilp_finish([], [], -1, output, 'Stopped by an output function before an integer feasible point was found.', lvl);
    elseif isempty(stopreason)
        [x, fval, exitflag, output] = ilp_finish([], [], -2, output, 'No integer feasible point exists.', lvl);
    else
        [x, fval, exitflag, output] = ilp_finish([], [], 0, output, sprintf('Stopped (%s) before finding an integer feasible point.', stopreason), lvl);
    end
else
    output.absolutegap = max(U - L, 0);
    output.relativegap = 100 * max(U - L, 0) / (abs(U) + 1);
    output.constrviolation = max([0; Ao * xbest - bo; abs(Aeq * xbest - beq); lb - xbest; xbest - ub]);
    if isempty(stopreason)
        ef = 1; msg = sprintf('Optimal solution found: the relative gap is %g%% (RelativeGapTolerance %g).', output.relativegap, rgap);
    else
        ef = 2; msg = sprintf('Stopped (%s) with an integer feasible point; relative gap %g%%.', stopreason, output.relativegap);
    end
    [x, fval, exitflag, output] = ilp_finish(xbest, f.' * xbest, ef, output, msg, lvl);
end
if OF.active, optim_outfcn(OF, x, ilp_ov(fval, nnodes, nfeas, output.relativegap, 'done'), 'done'); end
end

% ====================================================================== helpers

function tf = ilp_feasible(x, A, b, Aeq, beq, lb, ub, intcon, itol, ctol)
tf = all(abs(x(intcon) - round(x(intcon))) <= itol) && all(x >= lb - ctol) && all(x <= ub + ctol);
if tf && ~isempty(A), tf = all(A * x - b <= ctol * max(1, abs(b))); end
if tf && ~isempty(Aeq), tf = all(abs(Aeq * x - beq) <= ctol * max(1, abs(beq))); end
end

function [x, ok, basis] = ilp_lp(f, A, b, Aeq, beq, lb, ub, basis)
% LP relaxation of a node by the dual simplex, warm-started from the parent's basis.
% ok = 1 solved, 0 infeasible (or numerical failure).
x = [];
if any(lb > ub + 1e-9), ok = 0; return, end
[x, st, info] = optim_simplex(f, A, b, Aeq, beq, lb, ub, struct('basis', basis));
ok = double(st == 1);
if ok, basis = info.basis; x = min(max(x, lb), ub); end
end

function [Ac, bc] = ilp_gmi(info, A, b, Aeq, beq, intcon, n, maxcuts)
% Gomory mixed-integer cuts from the rows of the optimal simplex tableau whose basic variable is
% integer and fractional. Nonbasic variables are shifted to their active bounds; slacks of
% inequality rows with integer data on integer variables count as integer. A cut that uses a
% variable at an artificial bound is skipped. Returned as rows Ac*x <= bc.
Ac = zeros(0, n); bc = zeros(0, 1);
mA = size(A, 1); M = [A; Aeq]; m = size(M, 1);
Mf = [M, eye(m)];
B = info.basis.B; stat = info.basis.stat; xz = info.xz; la = info.la; ua = info.ua;
isint = false(n + m, 1); isint(intcon) = true;
for i = 1:mA
    nzc = find(A(i, :));
    isint(n + i) = all(isint(nzc)) && all(abs(A(i, nzc) - round(A(i, nzc))) < 1e-12) && abs(b(i) - round(b(i))) < 1e-12;
end
cand = find(isint(B));
fr = xz(B(cand)) - floor(xz(B(cand)));
keep = fr > 0.01 & fr < 0.99;
cand = cand(keep); fr = fr(keep);
[~, o] = sort(abs(fr - 0.5)); cand = cand(o);
rall = [b; beq];
MB = Mf(:, B); [L, U, P] = lu(MB);
for r = cand(:).'
    if size(Ac, 1) >= maxcuts, break, end
    er = zeros(m, 1); er(r) = 1;
    rho = P.' * (L.' \ (U.' \ er));
    alpha = (rho.' * Mf).';
    f0 = xz(B(r)) - floor(xz(B(r)));
    nb = find(stat ~= 0 & la ~= ua & abs(alpha) > 1e-11);
    if any(info.atart(nb)), continue, end
    a = alpha(nb); a(stat(nb) == 1) = -a(stat(nb) == 1);
    g = zeros(size(a));
    ii = isint(nb);
    fj = a(ii) - floor(a(ii));
    gi = fj / f0; up = fj > f0; gi(up) = (1 - fj(up)) / (1 - f0);
    g(ii) = gi;
    ac = a(~ii); gc = ac / f0; gc(ac < 0) = -ac(ac < 0) / (1 - f0);
    g(~ii) = gc;
    % sum g_j x'_j >= 1, back to x: x' = x - l (at lower), u - x (at upper), s = b - A x (slack)
    pi = zeros(n, 1); pi0 = 0;
    for t = 1:numel(nb)
        j = nb(t); gj = g(t);
        if gj == 0, continue, end
        if j <= n
            if stat(j) == -1, pi(j) = pi(j) + gj; pi0 = pi0 - gj * la(j);
            else, pi(j) = pi(j) - gj; pi0 = pi0 + gj * ua(j); end
        else
            row = j - n;                            % only inequality slacks are not fixed
            pi = pi - gj * M(row, :).'; pi0 = pi0 + gj * rall(row);
        end
    end
    % cut: pi'*x + pi0 >= 1  ->  -pi'*x <= pi0 - 1
    nzp = abs(pi(pi ~= 0));
    if isempty(nzp) || max(nzp) / min(nzp) > 1e6, continue, end
    sc = max(nzp);
    Ac(end + 1, :) = -pi.' / sc; bc(end + 1, 1) = (pi0 - 1) / sc; %#ok<AGROW>
end
end

function tf = ilp_prune(bound, U, agap, rgap, objint)
% a node with this LP bound cannot improve the incumbent U by more than the gap tolerances
tf = false;
if ~isfinite(U), return, end
tf = U - bound <= max(agap, rgap * (abs(U) + 1));
if ~tf && objint, tf = ceil(bound - 1e-6) >= U; end
end

function g = ilp_gap(U, L)
if ~isfinite(U), g = inf; else, g = 100 * max(U - L, 0) / (abs(U) + 1); end
end

function ov = ilp_ov(U, nnodes, nfeas, gap, phase)
ov = struct('phase', phase, 'fval', U, 'numnodes', nnodes, 'numfeaspoints', nfeas, ...
    'relativegap', gap, 'iteration', nnodes);
end

function [x, fval, ef, output] = ilp_finish(x, fval, ef, output, msg, lvl)
output.message = msg;
if lvl == 1 || lvl == 2 || (lvl == -1 && ef <= 0)
    fprintf('\n%s\n\n', msg);
end
end

function v = ilp_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end
