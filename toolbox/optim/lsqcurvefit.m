function [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqcurvefit(fun, x0, xdata, ydata, lb, ub, options)
%LSQCURVEFIT  Nonlinear curve fitting (data fitting) in the least-squares sense.
%   x = lsqcurvefit(fun, x0, xdata, ydata)          minimizes sum((fun(x, xdata) - ydata).^2)
%   x = lsqcurvefit(fun, x0, xdata, ydata, lb, ub)  with lb <= x <= ub
%   x = lsqcurvefit(fun, x0, xdata, ydata, lb, ub, options)
%   x = lsqcurvefit(problem)
%   [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqcurvefit(...)
%
%   residual = fun(x, xdata) - ydata. With SpecifyObjectiveGradient, fun returns [F, J] with
%   J = dF/dx. Solved by lsqnonlin (Levenberg-Marquardt with bounds).
%   Part of the MatLivre optimization toolbox draft (clean-room).
if nargin == 1 && isstruct(fun)
    p = fun; fun = p.objective; x0 = p.x0; xdata = p.xdata; ydata = p.ydata;
    lb = lcf_fld(p, 'lb'); ub = lcf_fld(p, 'ub'); options = lcf_fld(p, 'options');
    nin = 7;
else
    nin = nargin;
end
if nin < 5, lb = []; end
if nin < 6, ub = []; end
if nin < 7 || isempty(options), options = struct(); end
if ischar(fun) || isstring(fun), fun = str2func(fun); end
userJ = optim_getopt(options, {'SpecifyObjectiveGradient', 'Jacobian'}, false);
if ischar(userJ) || isstring(userJ), userJ = strcmpi(userJ, 'on'); end
yd = double(ydata);
if userJ
    res = @(x) lcf_withjac(fun, x, xdata, yd);
else
    res = @(x) reshape(fun(x, xdata) - yd, [], 1);
end
[x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(res, x0, lb, ub, options);
residual = reshape(residual, size(yd));
end

function [F, J] = lcf_withjac(fun, x, xdata, yd)
if nargout > 1
    [F, J] = fun(x, xdata);
else
    F = fun(x, xdata);
end
F = reshape(F - yd, [], 1);
end

function v = lcf_fld(s, name)
if isfield(s, name), v = s.(name); else, v = []; end
end
