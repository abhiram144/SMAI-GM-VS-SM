function [ cost ] = mutagen_etc(g1, ie1, g2, ie2)
% Returns the cost of transformation of edge ie1 of g1 into edge ie2 of g2
% for the mutagenicity database
%
% Input:
%
%   g1 - first graph
%   ie1 - indices of the ends of the source edge (to be transformed) in g1
%   g2 - second graph
%   ie2 - indices of the ends of the destination edge (transfrm into) in g2
%
% Output:
%
% Cost of transformation of ie1 into ie2. If ie1 == 0, then it is the cost
% of insertion of edge e2 into the graph g1. If ie2 == 0, then it is the
% cost of removal of edge ie1 from graph g1.
%
    deleting_ie1 = size(ie2, 2) == 1 && ie2 == 0;
    inserting_ie2 = size(ie1, 2) == 1 && ie1 == 0;
    if(deleting_ie1 && inserting_ie2)
        error('Both edges cannot be 0. Either delete or insert an edge.');
    end

    if(deleting_ie1)
        % cost of edge deletion
        cost = g1.adj(ie1(1),ie1(2));
    elseif(inserting_ie2)
        % cost of edge insertion
        cost = g2.adj(ie2(1), ie2(2));
    else
        % cost of edge substitution
        cost =  abs(g1.adj(ie1(1),ie1(2)) - g2.adh(ie2(1),ie2(2)));
    end
end