function [ refsets ] = get_refsets_simple(objs, ~, k)
%
% Selects k reference sets from objs using metric
%
% This particular implementation simply randomly picks
% k objects and designate each of them to be a singleton
% reference set. Metric (2'nd argument) is not used.
%
    n = length(objs);
    inds = randi([1; n], 1, k);
    refsets(k) = objs(1);
    for i = 1 : k
        ind = inds(i);
        refsets(i) = objs(ind);
    end
end