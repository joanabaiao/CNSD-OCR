% Computacao Neuronal e Sistemas Difusos 2020/21
% Andre Bernardes (2017248159) & Joana Baião (2017260526) - MIEB

% create_dataset: Juntar as 40 matrizes numa só matriz P e criar uma matriz
%                 T (target) com as mesmas dimensões. Criar as matrizes
%                 target para treino e para teste (T_train e T_test),
%                 respetivamente.

clear all;
clc;

n_matrix = 40;

P_total = [];
for i = 1:n_matrix
    load("P" + string(i) + ".mat");
    P_total = [P_total P];    
end

Perfect_total = [];
load("PerfectArial.mat");
for i = 1:5*n_matrix   
    Perfect_total = [Perfect_total Perfect];   
end

load("P_test1.mat")
P_test1 = P;

load("P_test2.mat")
P_test2 = P;

P = P_total; 
T = Perfect_total; 

T_train = repmat(diag(ones(1,10)), [1,200]);
T_test = repmat(diag(ones(1,10)), [1,5]);

save('dataset.mat', 'P', 'P_test1', 'P_test2', 'T', 'T_train', 'T_test');
