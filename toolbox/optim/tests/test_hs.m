if ~exist('fmincon', 'file'), d = fileparts(mfilename('fullpath')); if isempty(d), d = pwd; end, addpath(fullfile(d, '..')); end   % MatLivre: use this toolbox; MATLAB: keep its own solvers
% Hock-Schittkowski problems with known optima (HS6, 7, 13, 26, 39, 40, 43, 65, 100).
% Prints:  HS name ef f fstar |f-fstar| it fc viol
opts = optimoptions('fmincon', 'Display', 'off', 'MaxFunctionEvaluations', 5000, 'MaxIterations', 1000);
none = [];
P = {};
P{end+1} = {'HS6',  @(x) (1 - x(1))^2, [-1.2 1], none, none, @(x) deal([], 10*(x(2) - x(1)^2)), 0};
P{end+1} = {'HS7',  @(x) log(1 + x(1)^2) - x(2), [2 2], none, none, @(x) deal([], (1 + x(1)^2)^2 + x(2)^2 - 4), -sqrt(3)};
P{end+1} = {'HS13', @(x) (x(1) - 2)^2 + x(2)^2, [-2 -2], [0 0], none, @(x) deal(x(2) - (1 - x(1))^3, []), 1};
P{end+1} = {'HS26', @(x) (x(1) - x(2))^2 + (x(2) - x(3))^4, [-2.6 2 2], none, none, @(x) deal([], (1 + x(2)^2)*x(1) + x(3)^4 - 3), 0};
P{end+1} = {'HS39', @(x) -x(1), [2 2 2 2], none, none, @(x) deal([], [x(2) - x(1)^3 - x(3)^2; x(1)^2 - x(2) - x(4)^2]), -1};
P{end+1} = {'HS40', @(x) -x(1)*x(2)*x(3)*x(4), 0.8*ones(1, 4), none, none, ...
    @(x) deal([], [x(1)^3 + x(2)^2 - 1; x(1)^2*x(4) - x(3); x(4)^2 - x(2)]), -0.25};
P{end+1} = {'HS43', @(x) x(1)^2 + x(2)^2 + 2*x(3)^2 + x(4)^2 - 5*x(1) - 5*x(2) - 21*x(3) + 7*x(4), zeros(1, 4), none, none, ...
    @(x) deal(-[8 - x(1)^2 - x(2)^2 - x(3)^2 - x(4)^2 - x(1) + x(2) - x(3) + x(4);
                10 - x(1)^2 - 2*x(2)^2 - x(3)^2 - 2*x(4)^2 + x(1) + x(4);
                5 - 2*x(1)^2 - x(2)^2 - x(3)^2 - 2*x(1) + x(2) + x(4)], []), -44};
P{end+1} = {'HS65', @(x) (x(1) - x(2))^2 + (x(1) + x(2) - 10)^2/9 + (x(3) - 5)^2, [-5 5 0], [-4.5 -4.5 -5], [4.5 4.5 5], ...
    @(x) deal(x(1)^2 + x(2)^2 + x(3)^2 - 48, []), 0.9535288567};
P{end+1} = {'HS100', @(x) (x(1) - 10)^2 + 5*(x(2) - 12)^2 + x(3)^4 + 3*(x(4) - 11)^2 + 10*x(5)^6 + 7*x(6)^2 + x(7)^4 - 4*x(6)*x(7) - 10*x(6) - 8*x(7), ...
    [1 2 0 4 0 1 1], none, none, ...
    @(x) deal(-[127 - 2*x(1)^2 - 3*x(2)^4 - x(3) - 4*x(4)^2 - 5*x(5);
                282 - 7*x(1) - 3*x(2) - 10*x(3)^2 - x(4) + x(5);
                196 - 23*x(1) - x(2)^2 - 6*x(6)^2 + 8*x(7);
                -4*x(1)^2 - x(2)^2 + 3*x(1)*x(2) - 2*x(3)^2 - 5*x(6) + 11*x(7)], []), 680.6300573};
tall = tic;
for i = 1:numel(P)
    p = P{i};
    [x, f, ef, out] = fmincon(p{2}, p{3}, [], [], [], [], p{4}, p{5}, p{6}, opts);
    fprintf('HS %-5s ef=%2d f=%.10g fstar=%.10g err=%.2e it=%d fc=%d viol=%.1e\n', p{1}, ef, f, p{7}, abs(f - p{7}), ...
        out.iterations, out.funcCount, out.constrviolation);
end
fprintf('TOTAL %.3f s\n', toc(tall));
