function [ cost ] = letter_ntc(g1, n1, g2, n2)
% Returns the cost of node transform for the letter database
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
        cost = norm([n2.x, n2.y]);
    elseif(n2_is_empty)
        cost = norm([n1.x, n1.y]);
    else
        cost = norm([n1.x - n2.x, n1.y - n2.y]);
    end
end

