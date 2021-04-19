chems        = { 'Br',  'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'H', 'K', ...
                 'Na', 'Li', 'Ca' };
atomic_nums  = containers.Map(chems, ...
               {  35,    6,   17,   9,  53,   7,   8,  15,   16,  1,  19, ...
                  11,    3,   20});
n_valent_els = containers.Map(chems, ...
               {   7,    4,    7,   7,   7,   5,   6,   5,    6,  1,   1, ...
                   1,    1,    2 });

n = size(graphs, 2);
clear processed_graphs;
for i = 1 : n
    g = graphs(i);
    nnodes = size(g.v, 2);

    clear processed_v;
    for ia = 1 : nnodes
        atom = struct(g.v(ia));
        atom.atomic_number = atomic_nums(atom.chem);
        atom.n_valent_electrons = n_valent_els(atom.chem);
        if(exist('processed_v', 'var') == 0)
            processed_v(nnodes) = atom;
        end
        processed_v(ia) = struct(atom);
    end
    g.v = processed_v;

    g.shortest_paths = graphallshortestpaths(g.adj, 'directed', false);
    g.average_shpath_len = sum(g.shortest_paths, 2) / nnodes;

    if(exist('processed_graphs', 'var') == 0)
        processed_graphs(n) = g;
    end

    processed_graphs(i) = struct(g);
end
graphs = processed_graphs;