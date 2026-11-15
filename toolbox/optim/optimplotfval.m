function stop = optimplotfval(x, optimValues, state)
%OPTIMPLOTFVAL  Plot function: objective value at each iteration (for fsolve and the
%   least-squares solvers, the sum of squares). Use as PlotFcn in the toolbox solvers.
f = optimValues.fval;
if numel(f) > 1, f = sum(f(:).^2); end
stop = optim_plotscalar(optimValues.iteration, f, state, 'Current Function Value: %g', 'Function value');
end
