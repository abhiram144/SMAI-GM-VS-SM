failed_graph_idx = find(cat_mismatch_indicator);
nfailed = size(failed_graph_idx, 1);
if(nfailed > max_graphs_per_figure)
    sample_idx = randi([1 nfailed], [1 max_graphs_per_figure]);
    failed_graph_idx = failed_graph_idx(sample_idx);
end
letter_view_graphs(graphs(failed_graph_idx), 'Misclassified Graphs (Sample)');

succ_graph_idx = find(1 - cat_mismatch_indicator);
nsucc = size(succ_graph_idx, 1);
if(nsucc > max_graphs_per_figure)
    sample_idx = randi([1 nsucc], [1 max_graphs_per_figure]);
    succ_graph_idx = succ_graph_idx(sample_idx);
end
letter_view_graphs(graphs(succ_graph_idx), 'Well-Classified Graphs (Sample)');