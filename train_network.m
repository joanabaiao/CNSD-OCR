% Computacao Neuronal e Sistemas Difusos 2020/21
% Andre Bernardes (2017248159) & Joana Baião (2017260526) - MIEB

% train_network: Criar a rede neuronal e treina-la.

function [trained_net, train_results] = train_network(nn_architecture, activation_function, P, T_train)

% CRIAR NETWORK

if isequal(nn_architecture, 'AMC') || isequal(nn_architecture, 'BPC') || isequal(nn_architecture, 'C1') % 1 layer: apenas funcao de ativacao 
    net = network;

    net.numInputs = 1; 
    net.inputs{1}.size = 256;
    net.numLayers = 1; 
    net.layers{1}.size = 10;

    net.biasConnect(1) = 1;
    net.inputConnect(1) = 1;
    net.outputConnect(1) = 1;
    net.layers{1}.transferFcn = activation_function;

    W = rand(10,256);
    b = rand(10,1);
    net.IW{1,1} = W;
    net.b{1,1} = b;

    if isequal(activation_function, 'hardlim') % perceptron           
        net.trainFcn = 'trainc'; 
        net.inputWeights{1}.learnFcn = 'learnp';
        net.biases{1}.learnFcn = 'learnp';

    elseif isequal(activation_function, 'purelin') || isequal(activation_function, 'logsig') % gradiente
        net.trainFcn = 'trainlm'; 
        net.inputWeights{1}.learnFcn = 'trainlm';
        net.biases{1}.learnFcn = 'trainlm';    
    end

elseif isequal(nn_architecture, 'CS') % Classificador com softmax layer
            
    net = network;

    net.numInputs = 1; 
    net.inputs{1}.size = 256; 
    net.numLayers = 2;
    net.layers{1}.size = 13; 
    net.layers{2}.size = 10;

    net.biasConnect(1) = 1;
    net.biasConnect(2) = 1;
    net.inputConnect(1,1) = 1;
    net.layerConnect(2,1) = 1;
    net.outputConnect(2) = 1;

    net.layers{1}.name = 'Hidden Layer';
    net.layers{2}.name = 'Output Layer'; 

    net.layers{1}.transferFcn = activation_function;            
    net.layers{2}.transferFcn = 'softmax';

    net.trainFcn = 'trainlm'; 
    net.inputWeights{1,1}.learnFcn = 'trainlm';
    net.layerWeights{2,1}.learnFcn = 'trainlm';
    net.biases{1}.learnFcn = 'trainlm';
    net.biases{2}.learnFcn = 'trainlm';

    W1 = rand(13,256);
    W2 = rand(10,13);
    b1 = rand(13,1);
    b2 = rand(10,1);
    net.IW{1,1} = W1;
    net.b{1} = b1;
    net.LW{2,1} = W2;
    net.b{2} = b2;
    
    
elseif isequal(nn_architecture, 'C2') % Classificador com duas camadas
    
    activation_function1 = convertStringsToChars(activation_function(1));
    activation_function2 = convertStringsToChars(activation_function(2));
    
    net = network;
    net.numInputs = 1; 
    net.inputs{1}.size = 256; 
    net.numLayers = 2;
    net.layers{1}.size = 20; 
    net.layers{2}.size = 10;
    
    net.biasConnect(1) = 1;
    net.biasConnect(2) = 1;
    net.inputConnect(1,1) = 1;
    net.layerConnect(2,1) = 1;
    net.outputConnect(2) = 1;
    
    net.layers{1}.name = 'Hidden Layer';
    net.layers{2}.name = 'Output Layer';
    
    net.layers{1}.transferFcn = activation_function1;
    net.layers{2}.transferFcn = activation_function2;
    
    net.trainFcn = 'trainlm';    
    net.inputWeights{1,1}.learnFcn = 'trainlm';
    net.layerWeights{2,1}.learnFcn = 'trainlm';
    net.biases{1}.learnFcn = 'trainlm';
    net.biases{2}.learnFcn = 'trainlm';
    
    W1 = rand(20,256);
    W2 = rand(10,20);
    b1 = rand(20,1);
    b2 = rand(10,1);
    net.IW{1,1} = W1;
    net.b{1} = b1;
    net.LW{2,1} = W2;
    net.b{2} = b2;
   
end

% PARAMETERS
net.performParam.lr = 0.5;     % learning rate
net.trainParam.epochs = 1000;  % maximum epochs
net.trainParam.show = 35;      % show
net.trainParam.goal = 1e-6;    % goal = objective
net.performFcn = 'sse';        % criterion

net.divideFcn = 'dividerand';      % random division
net.divideMode = 'sample';
net.divideParam.trainRatio = 0.85; % training ratio
net.divideParam.valRatio = 0.15;   % validation ratio


% TREINO
net = configure(net, P, T_train);
[trained_net, training_records] = train(net, P, T_train);
train_results = trained_net(P);

figure
plotperf(training_records); title('Network performance');


end