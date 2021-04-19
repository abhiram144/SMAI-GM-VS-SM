% addpath data-letters
addpath libsvm-3.17/matlab
addpath geom2d/geom2d

ts_start = tic;

all_letters = {
      'A', 'E', 'F', 'H', 'I', ...
      'K', 'L', 'M', 'N', 'T', ... %
      'V', 'W', 'X', 'Y', 'Z' ... %
};
letters = all_letters;
per_letter_sample_size = Inf;
prob_being_in_training_set = 0.33;
dim = 32;

visualize = 1;
max_graphs_per_figure = 24;

disp('--> Loading graphs...'); tic;
letter_main_load_graphs;
letter_main_process_graphs;
fprintf('<-- Done in %fsec\n', toc);

disp('--> Performing Lipschitz embedding...'); tic;
V = lipschitz_embed(graphs, ...
    @get_refsets_simple, @min_dist_to_set, ...
    @(g1, g2) letter_graph_edit_dist(g1, g2, ...
        @letter_ntc_centroid, @letter_etc), ...
    dim);
fprintf('<-- Done in %fsec\n', toc);

V_cats = arrayfun(...
    @(fname) subsref(char(fname), struct('type','()','subs',{{1}})), ...
    {graphs.file} ...
);
V_cats = eval(strcat('{', sprintf('''%c''; ', V_cats), '}'));

V_ts_inds = get_training_set(V, prob_being_in_training_set);
V_ts = V(V_ts_inds, :);
cats_ts = V_cats(V_ts_inds, :);

disp('--> Classifying...'); tic;
disp('!!! SVMLIB output deliberately supressed');
output = evalc('computed_cats = classify_svm(V, V_ts, cats_ts)');
fprintf('<-- Done in %fsec\n', toc);

n = size(V, 1);
cat_diff = cell2mat(V_cats) - computed_cats;
cat_mismatch_indicator = cat_diff ~= 0;
accuracy = (n - sum(cat_mismatch_indicator)) / n;

if(visualize > 0)
    letter_main_view;
end

fprintf('!!! Classification accuracy = %.2f%%.\n', accuracy * 100);
toc(ts_start);