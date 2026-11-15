function stop = optimplotresnorm(x, optimValues, state)
%OPTIMPLOTRESNORM  Plot function: norm of the residuals (lsqnonlin, lsqcurvefit).
v = []; if isfield(optimValues, 'resnorm'), v = optimValues.resnorm; end
stop = optim_plotscalar(optimValues.iteration, v, state, 'Norm of Residuals: %g', 'Norm of residuals');
end
