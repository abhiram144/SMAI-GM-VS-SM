function [ d ] = letter_graph_edit_dist(g1, g2, ntc, etc)
% Computes graph edit distance from graph g1 to graph g2
%
% Input:
%
%   g1, g2 - graphs obtained through load_gxl_graph
%
%   ntc - handle to a node transform cost (ntc) function:
%         ntc(n1, n2) { return (cost of transform of n1 into n2) }
%         (structure of nodes n1, n2 is also defined by load_gxl_graph).
%         n2 or n1 may be 0 (corresponds to deletion or insertion, resp.)
%
%   etc - handle to an edge transform cost (etc) function:
%         TBD
% Output:
%
%   d - the (real) value of graph edit distance from g1 to g2
%

    n = length(g1.v);
    m = length(g2.v);

    g1_noutedges = full(sum(g1.adj ~= 0, 2));
    g2_noutedges = full(sum(g2.adj ~= 0, 2));

    edge_subs_costs = zeros(n + m, n + m);
    for i = 1 : n
        for j = 1 : m
            edge_subs_costs(i, j) = abs(g1_noutedges(i) - g2_noutedges(j));
        end
    end

    C = zeros(n + m, n + m);
    for i = 1 : n
        node_from = g1.v(i);
        for j = 1 : m
            node_to = g2.v(j);
            C(i, j) = ntc(g1, node_from, g2, node_to);
        end
    end
    C = C + edge_subs_costs;

    for i = 1 : n
        for j = m + 1 : n + m
            if(i == j - m)
                node_to_del = g1.v(i);
                C(i, j) = ntc(g1, node_to_del, g2, 0);
                C(i, j) = C(i, j) + g1_noutedges(i);
            else
                C(i, j) = Inf;
            end
        end
    end
    
    for j = 1 : m
        for i = n + 1 : n + m
            if(j == i - n)
                node_to_ins = g2.v(j);
                C(i, j) = ntc(g1, 0, g2, node_to_ins);
                C(i, j) = C(i, j) + g2_noutedges(j);
            else
                C(i, j) = Inf;
            end
        end
    end

    [~, d] = munkres(C); % ~ = optimal assignment
end