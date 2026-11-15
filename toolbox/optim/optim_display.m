function lvl = optim_display(options, default)
%OPTIM_DISPLAY  Display level from options: 0 off, 1 final, 2 iter, -1 notify (only on failure).
if nargin < 2, default = 'final'; end
d = lower(char(optim_getopt(options, {'Display'}, default)));
switch d
    case {'off', 'none'}, lvl = 0;
    case {'iter', 'iter-detailed'}, lvl = 2;
    case {'notify', 'notify-detailed'}, lvl = -1;
    otherwise, lvl = 1;
end
end
