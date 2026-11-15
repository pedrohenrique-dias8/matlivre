if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Tests for checkGradients. Prints CHECKGRAD lines.
% correct gradient
[v, e] = checkGradients(@tcg_rosen, [-1.2 1]);
fprintf('CHECKGRAD rosen_ok valid=%d maxerr=%.1e\n', v, max(e.Objective));
% wrong gradient (sign error in the second component)
[v, e] = checkGradients(@tcg_rosen_bad, [-1.2 1]);
fprintf('CHECKGRAD rosen_bad valid=%d maxerr=%.3g size=%s\n', v, max(e.Objective), mat2str(size(e.Objective)));
% Jacobian of a vector function (lsqnonlin / fsolve style)
[v, e] = checkGradients(@tcg_vec, [0.5 -0.3 2]);
fprintf('CHECKGRAD jacobian_ok valid=%d maxerr=%.1e size=%s\n', v, max(e.Objective(:)), mat2str(size(e.Objective)));
% nonlinear constraints with gradients (HS071), one wrong entry in the equality gradient
[v, e] = checkGradients(@tcg_hs71c, [1 5 5 1], 'IsConstraint', true);
fprintf('CHECKGRAD constraints_ok valid=%s\n', mat2str(v));
[v, e] = checkGradients(@tcg_hs71c_bad, [1 5 5 1], optimoptions('fmincon'), 'IsConstraint', true, 'Display', 'on');
fprintf('CHECKGRAD constraints_bad valid=%s maxeq=%.3g\n', mat2str(v), max(e.Equality(:)));
% matrix-shaped x0
[v, e] = checkGradients(@tcg_cubes, [1 2; 3 4], 'Tolerance', 1e-6);
fprintf('CHECKGRAD matrix_x0 valid=%d\n', v);

function [f, g] = tcg_cubes(X)
f = sum(X(:).^3); g = 3 * X.^2;
end
function [f, g] = tcg_rosen(x)
f = 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
g = [-400 * x(1) * (x(2) - x(1)^2) - 2 * (1 - x(1)); 200 * (x(2) - x(1)^2)];
end
function [f, g] = tcg_rosen_bad(x)
[f, g] = tcg_rosen(x); g(2) = -g(2);
end
function [F, J] = tcg_vec(x)
F = [x(1)^2 * x(2); sin(x(3)) + x(1); exp(x(2) * x(3))];
J = [2 * x(1) * x(2), x(1)^2, 0; 1, 0, cos(x(3)); 0, x(3) * exp(x(2) * x(3)), x(2) * exp(x(2) * x(3))];
end
function [c, ceq, gc, gceq] = tcg_hs71c(x)
c = 25 - prod(x); ceq = sum(x.^2) - 40;
gc = -[x(2)*x(3)*x(4); x(1)*x(3)*x(4); x(1)*x(2)*x(4); x(1)*x(2)*x(3)];
gceq = 2 * x(:);
end
function [c, ceq, gc, gceq] = tcg_hs71c_bad(x)
[c, ceq, gc, gceq] = tcg_hs71c(x); gceq(3) = x(3);
end
