function [ g ] = load_gxl_graph(filename)
% Builds a graph by its GXL description.
%
% Input: filename - the name of a GXL file
% Output: g - resulting graph (g.v - nodes, g.adj - adjacency matrix)
% Usage: g = load_gxl_graph('my_file.gxl')
%
    if(exist(filename, 'file') == 0)
        error(strcat('File "', filename, '" not found (forgot addpath?)'));
    end
    doc = xmlread(filename);
    
    gxl = find_unique_child_by_name(doc, 'gxl');
    if(gxl == -1)
        error('Invalid root. <gxl> is expected.');
    end

    graph = find_unique_child_by_name(gxl, 'graph');
    if(graph == -1)
        error('<graph> node not found.');
    end

    nodes = find_children_by_name(graph, 'node');

    for i = 1 : length(nodes)
        node = parse_node(nodes(i));
        if(exist('v', 'var') == 0)
            v(length(nodes)) = node;
        end
        v(i) = node;
    end

    node_idx = containers.Map({v.id}, 1:length({v.id}));
    n = node_idx.length();
    
    sparse_adj = 0;
    edges = find_children_by_name(graph, 'edge');
    for i = 1 : length(edges)
        edge = edges(i);
        from = char(edge.getAttribute('from'));
        from_ind = node_idx(from);
        to = char(edge.getAttribute('to'));
        to_ind = node_idx(to);

        w = struct();
        [w, nattrs, anames] = parse_node_attrs(edge, w);
        if(nattrs == 0)
            w = 1;
            sparse_adj = 1;
        elseif(nattrs == 1)
            w = w.(cell2mat(anames(1)));
            sparse_adj = 1;
        end
        if(exist('adj', 'var') == 0)
            if(nattrs > 2)
                dummy = struct();
                % TODO: will not work, since fields are not defined
            else
                dummy = 0;
            end;
            adj(n, n) = dummy;
        end
        adj(from_ind, to_ind) = w;
    end
    if(sparse_adj)
        adj = sparse(adj);
    end
    
    g = struct('file', filename, 'v', v, 'adj', adj);
end

function [ v ] = parse_node(node)
    v = struct();
    v.id = char(node.getAttribute('id'));
    v = parse_node_attrs(node, v);
end

function [new_node_obj, nattrs, anames] = parse_node_attrs(node, node_obj)
    new_node_obj = node_obj;
    attrs = find_children_by_name(node, 'attr');
    nattrs = length(attrs);
    anames = cell(nattrs);
    for i = 1 : nattrs
        attr = attrs(i);
        aname = char(attr.getAttribute('name'));
        anames(i) = {aname};
        aval_node = attr.getFirstChild();
        aval_node_name = aval_node.getNodeName();
        aval_str = char(aval_node.getFirstChild().getNodeValue());
        if(strcmp(aval_node_name, 'float'))
            aval = str2double(aval_str);
        elseif(strcmp(aval_node_name, 'int'))
            aval = round(str2double(aval_str));
        elseif(strcmp(aval_node_name, 'string'))
            aval = aval_str;
        else
            error('Unsupported attr value. Extend parse_node function.');
        end
        new_node_obj.(aname) = aval;
    end
end

function [ node ] = find_unique_child_by_name(root, name)
    nodes = find_children_by_name(root, name);
    if(isempty(nodes))
        node = -1;
    else
        node = nodes(1);
    end;
end

function [ nodes ] = find_children_by_name(root, name)
    nodes = [ ];

    if(root.getNodeType() == 10)
         children = root.getElements();
    else
         children = root.getChildNodes();
    end

    for i = 0 : children.getLength() - 1
        child = children.item(i);
        if(child.getNodeType() == 10)
            continue;
        end;
        if(strcmp(child.getNodeName(), name) == 1)
            nodes = [nodes, child];
        end
    end
end