function stop = optim_plotscalar(it, value, state, ttl, ylab, marker)
%OPTIM_PLOTSCALAR  Shared body of the optimplot*/xxplotbest* functions: one marker per
%   iteration on the current axes, the title showing the latest value (MatLivre toolbox).
stop = false;
if nargin < 6, marker = 'kd'; end
if isempty(value) || strcmp(state, 'done') || strcmp(state, 'interrupt'), return, end
ax = gca;
if strcmp(state, 'init')
    cla(ax);
    plot(ax, it, value, marker, 'MarkerFaceColor', [1 0 1]);
    hold(ax, 'on');
    xlabel(ax, 'Iteration'); ylabel(ax, ylab);
else
    plot(ax, it, value, marker, 'MarkerFaceColor', [1 0 1]);
end
title(ax, sprintf(ttl, value));
end
