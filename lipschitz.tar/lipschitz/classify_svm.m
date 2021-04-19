function [ cats ] = classify_svm(V, V_ts, cats_ts)
% Classify vectors v against the v_ts training set with categories cats_ts
%
% Input:
%
%   V - (row-)vectors to be classified
%   V_ts - training set (a few vectors with known categories)
%   cats_ts - categories of the vectors from V_ts
%
% Output: cats - vectors with a category for each row of V
%
% TODO: support more than 2 categories

% WARNING: Matlab's SVM classifier is license-conscious, and,
%          most likely, all (two) UCSB's licences for the hosting
%          Bioinformatics toolbox are already in use.
%
%   svmstruct = svmtrain(V_ts, cats_ts);
%   cats = svmclassify(svmstruct, V);

    num_cats_ts = double(cell2mat(cats_ts));
    model = svmtrain(num_cats_ts, V_ts);
    [num_cats, ~, ~] = svmpredict(zeros(size(V, 1), 1), V, model);
    cats = char(num_cats);
end