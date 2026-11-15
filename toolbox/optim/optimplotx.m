function stop = optimplotx(x, optimValues, state)
%OPTIMPLOTX  Plot function: bar chart of the current point.
stop = false;
if strcmp(state, 'done') || strcmp(state, 'interrupt'), return, end
ax = gca; cla(ax);
bar(ax, double(x(:)));
title(ax, 'Current Point'); xlabel(ax, sprintf('Number of variables: %d', numel(x))); ylabel(ax, 'Current point');
end
