load('mutagen_graphs.mat');

ngraphs_all = 4337;
ngraphs = round(dataset_part_to_use * ngraphs_all);
graphs = graphs(1:ngraphs);

% clear graphs;
% for i = 1 : ngraphs
%     filename = sprintf('molecule_%d.gxl', i);
%     if(~exist(filename, 'file'))
%         break;
%     end
% 
%     graph = load_gxl_graph(filename);
%     graph.adj = graph.adj + graph.adj';
%     if(i == 1)
%         graphs(ngraphs) = graph;
%     end
%     graphs(i) = graph;
% end