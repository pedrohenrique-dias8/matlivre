if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Validation problems for fmincon (MatLivre optimization toolbox draft).
% Prints one line per problem:  RESULT name exitflag f x... | iterations funcCount constrviolation
opts = optimoptions('fmincon', 'Display', 'off');

% P1 Rosenbrock inside the unit disk
rosen = @(x) 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
disk = @(x) deal(x(1)^2 + x(2)^2 - 1, []);
tic; [x, f, ef, out] = fmincon(rosen, [0 0], [], [], [], [], [], [], disk, opts); t = toc;
tfc_report('P1_rosen_disk', x, f, ef, out, t);

% P2 Hock-Schittkowski 71 (finite differences)
hs71 = @(x) x(1) * x(4) * (x(1) + x(2) + x(3)) + x(3);
hs71c = @(x) deal(25 - prod(x), sum(x.^2) - 40);
tic; [x, f, ef, out, lam] = fmincon(hs71, [1 5 5 1], [], [], [], [], ones(1, 4), 5 * ones(1, 4), hs71c, opts); t = toc;
tfc_report('P2_hs071_fd', x, f, ef, out, t);
fprintf('LAMBDA P2_hs071 ineqnonlin %.8f eqnonlin %.8f lower %s\n', lam.ineqnonlin, lam.eqnonlin, mat2str(lam.lower.', 6));

% P3 HS071 with analytic gradients
o3 = optimoptions(opts, 'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient', true);
tic; [x, f, ef, out] = fmincon(@tfc_hs71_fg, [1 5 5 1], [], [], [], [], ones(1, 4), 5 * ones(1, 4), @tfc_hs71_cg, o3); t = toc;
tfc_report('P3_hs071_grad', x, f, ef, out, t);

% P4 linear inequalities and bounds (convex quadratic)
q = @(x) (x(1) - 1)^2 + (x(2) - 2.5)^2;
A = [-1 2; 1 2; 1 -2]; b = [2; 6; 2];
tic; [x, f, ef, out, lam] = fmincon(q, [2 0], A, b, [], [], [0 0], [], [], opts); t = toc;
tfc_report('P4_linear', x, f, ef, out, t);
fprintf('LAMBDA P4_linear ineqlin %s\n', mat2str(lam.ineqlin.', 6));

% P5 bounds only: Rosenbrock in a box that excludes (1,1)
tic; [x, f, ef, out] = fmincon(rosen, [-1.2 1], [], [], [], [], [-2 -2], [0.8 2], [], opts); t = toc;
tfc_report('P5_bounds', x, f, ef, out, t);

% P6 nonlinear equality from an infeasible start
tic; [x, f, ef, out] = fmincon(@(x) x(1) + x(2), [1 0.5], [], [], [], [], [], [], @(x) deal([], x(1)^2 + x(2)^2 - 2), opts); t = toc;
tfc_report('P6_circle_eq', x, f, ef, out, t);

% P7 linear equality + nonlinear inequality, central differences
o7 = optimoptions('fmincon', 'Display', 'off', 'OptimalityTolerance', 1e-8, 'ConstraintTolerance', 1e-8, 'FiniteDifferenceType', 'central');
tic; [x, f, ef, out] = fmincon(@(x) exp(x(1)) * (4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1), [-1 1], ...
    [], [], [1 1], 0, [], [], @(x) deal([1.5 + x(1)*x(2) - x(1) - x(2); -x(1)*x(2) - 10], []), o7); t = toc;
tfc_report('P7_eq_central', x, f, ef, out, t);

% P8 matrix-shaped x0 through the problem struct
prob.objective = @(X) sum(sum((X - [1 2; 3 4]).^2));
prob.x0 = zeros(2); prob.Aineq = ones(1, 4); prob.bineq = 6;
prob.nonlcon = @(X) deal(X(1, 1)^2 + X(2, 2)^2 - 4, []);
prob.options = opts; prob.solver = 'fmincon';
tic; [x, f, ef, out] = fmincon(prob); t = toc;
tfc_report('P8_matrix_struct', x(:).', f, ef, out, t);
fprintf('SHAPE P8 %s\n', mat2str(size(x)));

% P9 infeasible problem
[x, f, ef, out] = fmincon(@(x) x(1)^2 + x(2)^2, [1 1], [], [], [], [], [], [], @(x) deal(x(1)^2 + 1, []), opts);
fprintf('RESULT P9_infeasible ef=%d viol=%.4g\n', ef, out.constrviolation);

% P10 Display iter on P1 (to see the table)
fprintf('--- iteration display (P1) ---\n');
fmincon(rosen, [0 0], [], [], [], [], [], [], disk, optimoptions('fmincon', 'Display', 'iter'));

function tfc_report(name, x, f, ef, out, t)
fprintf('RESULT %s ef=%d f=%.10g x=%s | it=%d fc=%d viol=%.2e fo=%.2e t=%.4fs\n', name, ef, f, ...
    mat2str(x(:).', 9), out.iterations, out.funcCount, out.constrviolation, out.firstorderopt, t);
end

function [f, g] = tfc_hs71_fg(x)
f = x(1) * x(4) * (x(1) + x(2) + x(3)) + x(3);
g = [x(4) * (2*x(1) + x(2) + x(3)); x(1) * x(4); x(1) * x(4) + 1; x(1) * (x(1) + x(2) + x(3))];
end

function [c, ceq, gc, gceq] = tfc_hs71_cg(x)
c = 25 - prod(x);
ceq = sum(x.^2) - 40;
gc = -[x(2)*x(3)*x(4); x(1)*x(3)*x(4); x(1)*x(2)*x(4); x(1)*x(2)*x(3)];
gceq = 2 * x(:);
end
