function [ cost ] = mutagen_ntc(g1, n1, g2, n2)
% Returns the cost of node transform for the mutagenicity database
%
% Input:
%
%   n1 - node from left graph
%   n2 - node from right graph
%
% Output:
%
% Cost of transformation of n1 into n2. If n1 == 0, then it is the cost
% of insertion of n2 into the left graph. If n2 == 0, then it is the cost
% of removal of n1 from the left graph.
%
    n1_is_empty = isreal(n1);
    n2_is_empty = isreal(n2);
    if(n1_is_empty && n2_is_empty)
        error('Both nodes cannot be empty.');
    end

    if(n1_is_empty)
        cost = bitshift(n2.n_valent_electrons, 6) + n2.atomic_number;
        % cost = 1; % n2.atomic_number; % cost = 1;
    elseif(n2_is_empty)
        cost = bitshift(n1.n_valent_electrons, 6) + n1.atomic_number;
        % cost = 1; % n1.atomic_number; % cost = 1;
    else
        nvalec_diff = abs(n1.n_valent_electrons - n2.n_valent_electrons);
        atomnum_diff = abs(n1.atomic_number - n2.atomic_number);
        cost = bitshift(nvalec_diff, 6) + atomnum_diff;
        % cost = 2; % n1.atomic_number + n2.atomic_number / 2;
    end
end