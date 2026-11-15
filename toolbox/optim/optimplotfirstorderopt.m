function stop = optimplotfirstorderopt(x, optimValues, state)
%OPTIMPLOTFIRSTORDEROPT  Plot function: first-order optimality measure at each iteration.
v = []; if isfield(optimValues, 'firstorderopt'), v = optimValues.firstorderopt; end
stop = optim_plotscalar(optimValues.iteration, v, state, 'First-order Optimality: %g', 'First-order optimality');
end
