function [stop, st, data] = optim_outfcn(st, varargin)
%OPTIM_OUTFCN  OutputFcn / PlotFcn support shared by the toolbox solvers (MatLivre toolbox).
%
%   st = optim_outfcn('setup', options)          local solvers: f(x, optimValues, state)
%   st = optim_outfcn('setup', options, 'pso')   particleswarm: f(optimValues, state)
%   st = optim_outfcn('setup', options, 'ps')    patternsearch: out(optimvalues, options, flag),
%                                                plot(optimvalues, flag)
%   st = optim_outfcn('setup', options, 'ga')    ga: [state, options, changed] = out(options,
%                                                state, flag), state = plot(options, state, flag)
%   [stop, st]       = optim_outfcn(st, x, optimValues, state)    (local)
%   [stop, st]       = optim_outfcn(st, optimValues, state)       (pso)
%   [stop, st]       = optim_outfcn(st, optimvalues, options, flag)   (ps)
%   [stop, st, gast] = optim_outfcn(st, options, gastate, flag)   (ga; stop when StopFlag is set)
%
%   OutputFcn and PlotFcn accept a function handle, a name, or a cell array of them. Plot
%   functions share one figure named 'Optimization Plot Function' (for ga, 'Genetic Algorithm'),
%   one subplot each; the matching axes are made current before each plot function is called.
%   An output or plot function that asks to stop makes the solver return exitflag -1.
if ischar(st) && strcmp(st, 'setup')
    options = varargin{1};
    mode = 'local'; if numel(varargin) >= 2, mode = varargin{2}; end
    st = struct('out', {oo_list(optim_getopt(options, {'OutputFcn', 'OutputFcns'}, []))}, ...
        'plot', {oo_list(optim_getopt(options, {'PlotFcn', 'PlotFcns'}, []))}, ...
        'fig', [], 'ax', [], 'mode', mode);
    st.active = ~isempty(st.out) || ~isempty(st.plot);
    stop = st; data = [];
    return
end
stop = false; data = [];
if ~st.active, return, end
switch st.mode
    case 'local', outargs = varargin(1:3); plotargs = outargs; state = varargin{3};
    case 'pso',   outargs = varargin(1:2); plotargs = outargs; state = varargin{2};
    case 'ps',    outargs = varargin(1:3); plotargs = varargin([1 3]); state = varargin{3};
    case 'ga',    outargs = varargin(1:3); plotargs = outargs; state = varargin{3}; data = varargin{2};
end
for i = 1:numel(st.out)
    switch st.mode
        case 'ga'
            data = st.out{i}(outargs{1}, data, state);
            stop = stop || (isfield(data, 'StopFlag') && ~isempty(data.StopFlag));
        otherwise
            stop = logical(st.out{i}(outargs{:})) || stop;
    end
end
np = numel(st.plot);
if np > 0
    if strcmp(state, 'init') || isempty(st.fig)
        if strcmp(st.mode, 'ga'), nm = 'Genetic Algorithm'; else, nm = 'Optimization Plot Function'; end
        st.fig = figure('Name', nm, 'NumberTitle', 'off');
        rows = ceil(sqrt(np)); cols = ceil(np / rows);
        st.ax = cell(1, np);
        for k = 1:np, st.ax{k} = subplot(rows, cols, k); end
    end
    for k = 1:np
        axes(st.ax{k});
        if strcmp(st.mode, 'ga')
            data = st.plot{k}(plotargs{1}, data, state);
            stop = stop || (isfield(data, 'StopFlag') && ~isempty(data.StopFlag));
        else
            stop = logical(st.plot{k}(plotargs{:})) || stop;
        end
    end
    drawnow;
end
end

function c = oo_list(v)
if isempty(v), c = {}; return, end
if ~iscell(v), v = {v}; end
c = cell(1, numel(v));
for i = 1:numel(v)
    f = v{i};
    if ischar(f) || isstring(f), f = str2func(char(f)); end
    c{i} = f;
end
end
