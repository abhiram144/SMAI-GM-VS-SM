function [ts_inds] = select_training_set(objs, pr_in_ts)
% Selects a few objects from objs to be in a training set
%
% Input:
%
%   objs - all objects,
%   pr_in_ts - probability that an individual object will be selected
%              to the training set
%
% Output: ts_inds - indices of objects selected to the training set
%
    n = size(objs, 1);
    rnds = rand(n, 1);
    ts_inds = find(rnds < pr_in_ts);
end