% addpath data-mutagen
addpath libsvm-3.17/matlab
addpath geom2d/geom2d
addpath mit-mtna

% -- DEBUG ---------------------------
clear ged_cache;
global ged_cache;
ged_cache = -1 * ones(5000, 5000);
% ------------------------------------

ts_start = tic;

dataset_part_to_use = 1; % 1.0 = whole set (~4.5k molecules)
prob_being_in_training_set = 0.33;
dim = 32; % 32

disp('--> Loading graphs...'); tic;
mutagen_main_load_graphs;
% mutagen_main_process_graphs; % loaded already processed
mutagen_main_load_cats;
fprintf('<-- Done in %fsec\n', toc);

disp('--> Performing Lipschitz embedding...'); tic;
V = lipschitz_embed(graphs, ...
    @get_refsets_advanced, @min_dist_to_set, ...
    @(g1, g2) mutagen_graph_edit_dist(g1, g2, @mutagen_ntc, @mutagen_etc), ...
    dim);
fprintf('<-- Done in %fsec\n', toc);

V_ts_inds = get_training_set(V, prob_being_in_training_set);
V_ts = V(V_ts_inds, :);
cats_ts = cats(V_ts_inds);

disp('--> Classifying...'); tic;
disp('!!! SVMLIB output deliberately supressed');
output = evalc('computed_cats = classify_svm(V, V_ts, cats_ts)');
fprintf('<-- Done in %fsec\n', toc);

n = size(V, 1);
cat_diff = cell2mat(cats) - computed_cats;
cat_mismatch_indicator = cat_diff ~= 0;
accuracy = (n - sum(cat_mismatch_indicator)) / n;

fprintf('!!! Classification accuracy = %.2f%%.\n', accuracy * 100);
toc(ts_start);

save('last_session');