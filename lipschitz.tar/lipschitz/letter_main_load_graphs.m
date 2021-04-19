load('letter_graphs.mat');

if(per_letter_sample_size < Inf)
    nletters_all = size(all_letters, 2);
    nletters = size(letters, 2);
    ngraphs = size(graphs, 2);
    graphs = reshape(graphs, ngraphs / nletters_all, nletters_all);
    graphs = graphs(1 : per_letter_sample_size, 1 : nletters);
    graphs = reshape(graphs, 1, nletters * per_letter_sample_size);
end

% graphs = [];
% graphs_per_letter = zeros(1, length(letters));
% 
% for i = 1 : length(letters)
%     letter = cell2mat(letters(i));
%     count = 0;
%     graphs_per_letter(i) = 0;
%     while(1)
%         filename = sprintf('%sP1_%.4d.gxl', letter, count);
%         if(~exist(filename, 'file'))
%             break;
%         end
% 
%         graph = load_gxl_graph(filename);
%         graphs = [graphs, graph];
%         graphs_per_letter(i) = graphs_per_letter(i) + 1;
%         
%         count = count + 1;
%         if(count >= per_letter_sample_size)
%             break;
%         end
%     end
% end