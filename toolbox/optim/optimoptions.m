function o = optimoptions(solver, varargin)
%OPTIMOPTIONS  Options for optimization solvers (MatLivre optimization toolbox draft).
%   o = optimoptions('fmincon', 'Name', value, ...)
%   o = optimoptions(o, 'Name', value, ...)
%   o = optimoptions(@fmincon, ...)
%   Returns a struct with the field SolverName plus every option given; solvers read option
%   names in either the optimoptions style (MaxIterations) or the optimset style (MaxIter).
if isstruct(solver)
    o = solver;
else
    if isa(solver, 'function_handle'), solver = func2str(solver); end
    o = struct('SolverName', char(solver));
end
if mod(numel(varargin), 2) ~= 0
    error('MatLivre:optimoptions:pairs', 'Options must be given as name-value pairs.');
end
for i = 1:2:numel(varargin)
    o.(char(varargin{i})) = varargin{i + 1};
end
end
