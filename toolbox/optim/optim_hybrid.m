function [x, fval, flag, nev] = optim_hybrid(hyb, fun, x, fval, lb, ub, A, b, Aeq, beq, nonlcon)
%OPTIM_HYBRID  Run the HybridFcn of a global solver from its best point (MatLivre toolbox).
%   hyb is a name or handle (fmincon, fminunc, fminsearch, patternsearch) or {fcn, options}.
%   The local result replaces x only if it is better (and, for fmincon, feasible).
hopts = [];
if iscell(hyb), hopts = hyb{2}; hyb = hyb{1}; end
if isa(hyb, 'function_handle'), hyb = func2str(hyb); end
hyb = char(hyb);
if isempty(hopts), hopts = struct('Display', 'off'); end
if isstruct(hopts) && ~isfield(hopts, 'Display'), hopts.Display = 'off'; end
flag = []; nev = 0;
switch lower(hyb)
    case 'fmincon'
        [xh, fh, flag, out] = fmincon(fun, x, A, b, Aeq, beq, lb, ub, nonlcon, hopts);
        nev = out.funcCount;
        ok = flag > 0 || out.constrviolation <= 1e-6;
        if flag > 0, fval = inf; end   % a converged constrained local solution replaces the global one
    case 'fminunc'
        [xh, fh, flag, out] = fminunc(fun, x, hopts);
        nev = out.funcCount;
        ok = all(xh >= lb) && all(xh <= ub);
    case 'fminsearch'
        [xh, fh, flag, out] = fminsearch(fun, x, struct('Display', 'off'));
        nev = out.funcCount;
        ok = all(xh >= lb) && all(xh <= ub);
    case 'patternsearch'
        [xh, fh, flag, out] = patternsearch(fun, x, A, b, Aeq, beq, lb, ub, nonlcon, hopts);
        nev = out.funccount;
        ok = flag > 0;
    otherwise
        error('MatLivre:optim:hybrid', 'Unknown HybridFcn ''%s''.', hyb);
end
if ok && fh < fval
    x = reshape(xh, size(x)); fval = fh;
end
end
