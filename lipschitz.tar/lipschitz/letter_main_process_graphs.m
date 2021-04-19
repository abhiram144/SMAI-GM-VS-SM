n = size(graphs, 2);
for i = 1 : n
    metrics = letter_comp_graph_metrics(graphs(i).v, graphs(i).adj);
    graphs(i).metrics = metrics;
end