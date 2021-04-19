function [ d ] = min_dist_to_set(obj, set, metric)
% Computes the min distance d from obj to a set using metric
    if(iscell(set) > 0)
        dists = arrayfun(@(setobj) metric(obj, setobj), cell2mat(set));
    else
        dists = arrayfun(@(setobj) metric(obj, setobj), set);
    end
    d = min(dists);
end