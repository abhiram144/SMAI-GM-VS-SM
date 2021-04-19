function letter_view_graphs(gs, title_str)
    figure('name', title_str);
    ngs = size(gs, 2);
    rc = numSubplots(ngs);
    for ig = 1 : ngs
        g = gs(ig);
        n = size(g.v, 2);
        X = [g.v.x];
        Y = [g.v.y];
        subplot(rc(1), rc(2), ig);
        for i = 1 : n
            for j = i + 1 : n
                if(g.adj(i, j) > 0)
                    plot([X(i) X(j)], [Y(i) Y(j)], '-');
                    hold on;
                end
            end
        end

        if(isfield(g, 'metrics') > 0 && isfield(g.metrics, 'centroid') > 0)
            cent = g.metrics.centroid;
            plot(cent(1), cent(2), '*r');
        end

        title(g.file);
    end
end