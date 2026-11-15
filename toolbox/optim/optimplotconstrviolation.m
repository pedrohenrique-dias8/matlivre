function stop = optimplotconstrviolation(x, optimValues, state)
%OPTIMPLOTCONSTRVIOLATION  Plot function: maximum constraint violation at each iteration.
v = []; if isfield(optimValues, 'constrviolation'), v = optimValues.constrviolation; end
stop = optim_plotscalar(optimValues.iteration, v, state, 'Max Constraint Violation: %g', 'Constraint violation');
end
