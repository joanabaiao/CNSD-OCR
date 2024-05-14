% Computacao Neuronal e Sistemas Difusos 2020/21
% Andre Bernardes (2017248159) & Joana Baião (2017260526) - MIEB

% myclassify: carrega as redes neuronais já treinadas guardadas e testa​
%             os ​dados​ ​introduzidos na grelha.

function predicted = myclassify(data, filled_ind) 

global choice1 choice2;
nn_architecture = choice1;
activation_function = choice2;

if isequal(nn_architecture, 'AMC') % Associative memory + Classifier
    
    filename = strcat('NN_AMC_',activation_function,'.mat');   
    
    load(filename, 'trained_net');   
    load('W', 'W');
    data = W * data;
    
elseif isequal(nn_architecture, 'BPC') % Binary perceptron memory + Classifier
    
    filename = strcat('NN_BPC_',activation_function,'.mat');
    
    load(filename, 'trained_net');  
    load('perceptron_filter.mat', 'trained_filter');
    data = sim(trained_filter ,data);
    
elseif isequal(nn_architecture, 'C1') || isequal(nn_architecture, 'CS') % Classifier with one layer
    
    filename = strcat('NN_', nn_architecture, '_',activation_function,'.mat');    
    load(filename, 'trained_net');
    
elseif isequal(nn_architecture, 'C2') % Classifier with two layers
     
    filename = strcat('NN_C2_',activation_function(1),'_',activation_function(2),'.mat');   
    load(filename, 'trained_net');
    
end

test_results = sim(trained_net, data);
[~, ind] = max(test_results);
predicted = ind(filled_ind);

end
