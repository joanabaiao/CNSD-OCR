% Computacao Neuronal e Sistemas Difusos 2020/21
% Andre Bernardes (2017248159) & Joana Baiao (2017260526) - MIEB

% main: funcao principal que é chamada na interface

function main(nn_architecture, activation_function, choice)

disp(nn_architecture);
disp(activation_function);
disp(choice);
        
global choice1 choice2;
choice1 = nn_architecture;
choice2 = activation_function;
    
% LOAD AND CREATE NECESSARY MATRICES
load('dataset.mat', 'P', 'P_test1', 'P_test2', 'T', 'T_train', 'T_test'); 

% TEST TRAINED NETWORK
if isequal(choice, 'test')   
    mpaper;

% TRAIN NETWORK AND TEST
elseif isequal(choice, 'train')
    
    W = [];
    
    % FILTER
    if isequal(nn_architecture, 'AMC') % Associative memory as filter
        W = T * pinv(P); % Pseudo-inverse method: Wp = T * pinv(P)
        save('W.mat', 'W');
        
        P = W * P; % Output: P2 --> P1
        P_test1 = W * P_test1; 
        P_test2 = W * P_test2; 

    elseif isequal(nn_architecture, 'BPC')  % Binary perceptron as filter
        
        perceptron_filter = perceptron('hardlim', 'learnp');
        perceptron_filter.trainFcn = 'trainc';
   
        perceptron_filter.performParam.lr = 0.5;
        perceptron_filter.trainParam.epochs = 1000;
        perceptron_filter.trainParam.show = 35;
        perceptron_filter.trainParam.goal = 1e-6;
        perceptron_filter.performFcn = 'sse';

        perceptron_filter.divideFcn = 'dividerand';
        perceptron_filter.divideMode = 'sample';
        perceptron_filter.divideParam.trainRatio = 0.85;
        perceptron_filter.divideParam.valRatio = 0.15;
   
        [trained_filter, ~] = train(perceptron_filter, P, T);
        save ('perceptron_filter.mat', 'trained_filter');
        
        P = sim(trained_filter, P); %Output: P2 --> P1
        P_test1 = sim(trained_filter, P_test1);
        P_test2 = sim(trained_filter, P_test2);        
    end
    
    % TRAINING   
    [trained_net, train_results] = train_network(nn_architecture, activation_function, P, T_train);
    
    % Post-processing - Heuristic
    if isequal(nn_architecture, 'C2')       
        train_results = heuristic(train_results);
        filename = strcat('NN_C2_',activation_function(1),'_',activation_function(2),'.mat');  
 
    else
        filename = strcat('NN_',nn_architecture,'_',activation_function,'.mat');
        if isequal(activation_function, 'logsig') || isequal(activation_function, 'purelin')
            train_results = heuristic(train_results);
        end  
    end
    
    save(filename, 'trained_net');
    
    % VALIDATION                
    test_results1 = sim(trained_net, P_test1);
    test_results2 = sim(trained_net, P_test2);
          
    % ROC
    figure
    plotroc(T_train, train_results); title('ROC - Train');

    figure
    plotroc(T_test, test_results1); title('ROC - Test 1');

    figure
    plotroc(T_test, test_results2); title('ROC - Test 2');

    
    % CONFUSION MATRIX
    figure
    plotconfusion(T_train, train_results); title('Confusion Matrix - Train');

    figure
    plotconfusion(T_test, test_results1); title('Confusion Matrix - Test 1');

    figure
    plotconfusion(T_test, test_results2); title('Confusion Matrix - Test 2');
    
     
end

end

