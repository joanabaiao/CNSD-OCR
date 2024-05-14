% Computacao Neuronal e Sistemas Difusos 2020/21
% Andre Bernardes (2017248159) & Joana Baião (2017260526) - MIEB

% heuristic: A função heuristica muda o valor máximo de cada coluna para 1
%            e os restantes para 0.

function new_results = heuristic(results)

[~, ind] = max(results);
new_results = zeros(size(results));

for i = 1:size(results,2)
   new_results(ind(i), i)=1; 
end
    
end

