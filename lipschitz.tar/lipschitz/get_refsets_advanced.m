function [ refsets ] = get_refsets_advanced(objs, ~, k)
%
% Selects k reference sets from objs using metric
%
% This implementation is supposed to include multiple
% elements in each of k reference sets.
    n = length(objs);
    refset_size = 3;
    inds = randi([1; n], refset_size, k);
    refsets(k) = {objs(1, 1:refset_size)};
    for i = 1 : k
        ind = inds(:, i);
        refsets(i) = {objs(ind)};
    end
end