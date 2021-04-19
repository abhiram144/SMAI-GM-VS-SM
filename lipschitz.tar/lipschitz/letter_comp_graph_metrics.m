function [ metrics ] = letter_comp_graph_metrics(v, adj)
% COMP_GRAPH_METRICS Computes any metrics of a graph necessary for
% the further processing of this graph (e.g., all-to-all shortest paths,
% central vertices, ...). Will be stored in g.metrics.

    % cent = polygonCentroid([v.x], [v.y]);
    points = [[v.x]', [v.y]'];
    cent = centroid(points);
    metrics = struct('centroid', cent);

end