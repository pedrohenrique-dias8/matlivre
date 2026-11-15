function v = optim_getopt(s, names, default)
%OPTIM_GETOPT  Read an option from an optimoptions/optimset struct (MatLivre optimization toolbox).
%   v = optim_getopt(options, {'MaxIterations', 'MaxIter'}, default) returns the first non-empty
%   field among the given names, or default.
v = default;
if ~isstruct(s), return, end
if ischar(names), names = {names}; end
for i = 1:numel(names)
    if isfield(s, names{i}) && ~isempty(s.(names{i}))
        v = s.(names{i});
        return
    end
end
end
