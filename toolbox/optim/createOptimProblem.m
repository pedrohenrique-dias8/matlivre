function problem = createOptimProblem(solver, varargin)
%CREATEOPTIMPROBLEM  Create a problem structure for MultiStart or GlobalSearch.
%   problem = createOptimProblem('fmincon', 'objective', fun, 'x0', x0, 'lb', lb, 'ub', ub, ...)
%   Solvers: 'fmincon', 'fminunc', 'lsqnonlin', 'lsqcurvefit'. Names (case-insensitive):
%   objective, x0, Aineq, bineq, Aeq, beq, lb, ub, nonlcon, xdata, ydata, options.
%   Part of the MatLivre optimization toolbox draft (clean-room).
solver = char(solver);
valid = {'fmincon', 'fminunc', 'lsqnonlin', 'lsqcurvefit'};
if ~any(strcmpi(solver, valid))
    error('MatLivre:createOptimProblem:solver', 'Solver must be one of: %s.', strjoin(valid, ', '));
end
problem = struct('solver', lower(solver), 'objective', [], 'x0', [], 'Aineq', [], 'bineq', [], ...
    'Aeq', [], 'beq', [], 'lb', [], 'ub', [], 'nonlcon', [], 'options', struct('Display', 'off'));
if any(strcmpi(solver, {'lsqcurvefit'})), problem.xdata = []; problem.ydata = []; end
names = fieldnames(problem);
if mod(numel(varargin), 2) ~= 0
    error('MatLivre:createOptimProblem:pairs', 'Arguments after the solver must be name-value pairs.');
end
for i = 1:2:numel(varargin)
    k = find(strcmpi(char(varargin{i}), names), 1);
    if isempty(k)
        error('MatLivre:createOptimProblem:name', 'Unknown problem field ''%s''.', char(varargin{i}));
    end
    problem.(names{k}) = varargin{i + 1};
end
if isempty(problem.objective) || isempty(problem.x0)
    error('MatLivre:createOptimProblem:required', 'A problem needs at least ''objective'' and ''x0''.');
end
end
