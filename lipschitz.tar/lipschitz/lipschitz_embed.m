function [ V ] = lipschitz_embed(objs, get_refsets, dist_to_set, metric, d)
% Embeds metric space objects into a d-dimensional vector space
% using Lipschitz embedding.
%
% Input:
%
%   objs - array of objects (e.g., graphs)
%
%   get_refsets - handle to a function that creates reference sets
%
%       get_refs(objs, metric, k) { return (k subsets of objs) }
%
%       For example, each of k subsets may contain only one
%       randomly chosen object. More sophisticated strategies are
%       [Linial-London-Rabinovich'95] (random subsets of var. size)
%       and [Riesen-Bunke'09] (k-medoids clustering-based selection).
%
%   dist_to_set - handle to function that computes distance from obj to a set
%
%       dist_to_set(obj, set, metric) { return (distance from obj to set) }
%       
%       This can be min{norm(obj - *) | taken by all * from set}, as well
%       as max, or avg.
%
%   metric - handle to a metric function that computes distances btw objs
%
%   d - dimensionality of the embedding
%
% Output: V - matrix of vectorized objs; each row corresponds to an object
%
    n = length(objs);
    V = zeros(n, d);
    refsets = get_refsets(objs, metric, d);
    
    percent = round(n * 5 / 100);
    count = 0;

    for i = 1 : n
        obj = objs(i);
        for k = 1 : d
            V(i,k) = dist_to_set(obj, refsets(k), metric);
        end
        if(mod(i, percent) == 0)
            if(i == 1)
                fprintf(2, 'Done: ');
            end
            fprintf(2, '%d%%|', 5 * count);
            count = count + 1;
            if(i == n)
                fprintf(2, '\n');
            end
        end
    end

end