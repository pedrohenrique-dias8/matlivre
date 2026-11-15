function [x, yeq, zin, info] = optim_qpip(H, f, Ai, bi, Ae, be, opts)
%OPTIM_QPIP  Dense convex QP by a primal-dual interior-point method (Mehrotra predictor-corrector).
%
%   minimize    0.5*x'*H*x + f'*x
%   subject to  Ai*x <= bi,   Ae*x = be
%
%   H must be symmetric positive semidefinite. Returns the primal solution x, the equality
%   multipliers yeq, the inequality multipliers zin >= 0 (Lagrangian L = q + yeq'*(Ae*x-be) +
%   zin'*(Ai*x-bi)) and info with fields .converged, .iterations, .resid.
%
%   Reference: J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed., Sec. 16.6.
%   Written from the algorithm description; part of the MatLivre optimization toolbox draft.

n = numel(f); f = f(:);
if nargin < 3 || isempty(Ai), Ai = zeros(0, n); bi = zeros(0, 1); end
if nargin < 5 || isempty(Ae), Ae = zeros(0, n); be = zeros(0, 1); end
if nargin < 7, opts = struct(); end
tol = qpip_getopt(opts, 'tol', 1e-10);
maxit = qpip_getopt(opts, 'maxit', 200);
bi = bi(:); be = be(:);
mi = size(Ai, 1); me = size(Ae, 1);
H = (H + H.') / 2;
regH = 1e-12 * max(1, norm(H, 1));
regE = 1e-12;

% starting point: equality-constrained least-norm x, slacks and duals bounded away from zero
x = zeros(n, 1);
yeq = zeros(me, 1);
s = max(1, bi - Ai * x);
zin = ones(mi, 1);

scale = 1 + max([norm(f, inf), norm(bi, inf), norm(be, inf), norm(H, inf)]);
regH = max(regH, 1e-10 * scale);
info.converged = false;
info.resid = inf;
% Mehrotra-type starting point (Nocedal & Wright, p. 484): one affine step from x = 0, s = z = 1,
% then move the slacks and multipliers back into the interior.
if mi > 0
    w = zin ./ s;
    [Lk, Uk, Pk] = qpip_factor_kkt(H, Ai, Ae, w, regH, regE);
    rd = H * x + f + Ai.' * zin + Ae.' * yeq; re = Ae * x - be; ri = Ai * x + s - bi;
    if ~isempty(Lk)
        [dx, dy, ds, dz] = qpip_kkt_step(Lk, Uk, Pk, Ai, s, zin, rd, re, ri, -s .* zin, n);
    else
        dx = NaN;
    end
    if all(isfinite(dx)) && all(isfinite([dy; ds; dz]))
        x = x + dx; yeq = yeq + dy;
        s = max(1, abs(s + ds)); zin = max(1, abs(zin + dz));
    end
end
xb = x; yb = yeq; zb = zin; resb = inf; rpb = inf; rdb = inf; mub = inf;
it = 0;
for it = 1:maxit
    rd = H * x + f + Ai.' * zin + Ae.' * yeq;
    re = Ae * x - be;
    ri = Ai * x + s - bi;
    if mi > 0, mu = (s.' * zin) / mi; else, mu = 0; end
    res = max([norm(rd, inf), norm(re, inf), norm(ri, inf), mu]);
    if ~isfinite(res), break, end
    if res < resb, xb = x; yb = yeq; zb = zin; resb = res; rpb = max([norm(re, inf), norm(ri, inf), 0]); rdb = norm(rd, inf); mub = mu; end
    xmax = norm(x, inf);
    if xmax > 1e12 * scale, break, end   % diverging: unbounded problem (or no interior)
    if res <= tol * scale
        info.converged = true;
        break
    end
    w = zin ./ s;
    [Lk, Uk, Pk] = qpip_factor_kkt(H, Ai, Ae, w, regH, regE);
    if isempty(Lk), break, end
    % predictor (affine scaling)
    rc = -s .* zin;
    [dx, dy, ds, dz] = qpip_kkt_step(Lk, Uk, Pk, Ai, s, zin, rd, re, ri, rc, n);
    ap = qpip_step_to_boundary(s, ds, 1);
    ad = qpip_step_to_boundary(zin, dz, 1);
    if mi > 0
        mu_aff = ((s + ap * ds).' * (zin + ad * dz)) / mi;
        sigma = (mu_aff / mu)^3;
        % corrector
        rc = -s .* zin - ds .* dz + sigma * mu;
        [dx, dy, ds, dz] = qpip_kkt_step(Lk, Uk, Pk, Ai, s, zin, rd, re, ri, rc, n);
    end
    ap = qpip_step_to_boundary(s, ds, 0.995);
    ad = qpip_step_to_boundary(zin, dz, 0.995);
    if ~all(isfinite([dx; dy; ds; dz])), break, end
    x = x + ap * dx; s = s + ap * ds;
    yeq = yeq + ad * dy; zin = zin + ad * dz;
end
xlast = norm(x, inf);
% return the best iterate seen (the last one unless the method stalled or broke down)
x = xb; yeq = yb; zin = zb;
info.resid = resb;
info.primal = rpb;      % primal infeasibility at the returned point
info.dual = rdb;        % dual infeasibility
info.mu = mub;          % complementarity
info.scale = scale;
info.diverged = xlast > 1e8 * scale || ~isfinite(xlast);
info.iterations = it;
end

function [Lk, Uk, Pk] = qpip_factor_kkt(H, Ai, Ae, w, regH, regE)
% LU of the reduced KKT matrix after symmetric diagonal equilibration. The barrier terms make the
% matrix badly scaled near the solution; scaling by sqrt(|diag|) keeps the factors well
% conditioned. The scaling is folded into the factors: K = D*Ks*D, Ks = Pk'*Lk*Uk.
n = size(H, 1); me = size(Ae, 1);
Hw = H + Ai.' * (w .* Ai);
for tries = 1:6
    K = [Hw + regH * eye(n), Ae.'; Ae, -regE * eye(me)];
    dg = sqrt(max(abs(diag(K)), 1e-12));
    Ks = K ./ (dg * dg.');
    [Lk, Uk, Pk] = lu(Ks);
    if rcond(Uk) >= eps, break, end
    regH = regH * 1e3; regE = regE * 1e3;   % numerically singular: regularise more and retry
end
if rcond(Uk) < eps
    Lk = [];   % still singular: signal a breakdown to the caller
    return
end
Lk = struct('L', Lk, 'd', dg);
end

function [dx, dy, ds, dz] = qpip_kkt_step(Lk, Uk, Pk, Ai, s, z, rd, re, ri, rc, n)
% Newton step of the perturbed KKT system, with ds and dz eliminated.
r1 = -rd - Ai.' * ((rc + z .* ri) ./ s);
dg = Lk.d;
sol = (Uk \ (Lk.L \ (Pk * ([r1; -re] ./ dg)))) ./ dg;
dx = sol(1:n);
dy = sol(n+1:end);
ds = -ri - Ai * dx;
dz = (rc - z .* ds) ./ s;
end

function a = qpip_step_to_boundary(v, dv, frac)
neg = dv < 0;
if any(neg)
    a = min(1, frac * min(-v(neg) ./ dv(neg)));
else
    a = 1;
end
end

function v = qpip_getopt(s, name, default)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end
