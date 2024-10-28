clear;clc;close all;
Mdir = pwd;
Root = fileparts(Mdir);
addpath('./utils');
net_name = 'Resnet_101';
responseName = 'ANN_response_non_ori.mat';
results_path = fullfile('ANN_output/', net_name);
if ~exist(results_path,'dir')
    mkdir(results_path)
end

%% open pool
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    poolobj=parpool('local',4);
end

%% load data
full_path = fullfile(results_path, responseName);
load(full_path, 'ANN_response_non_ori')
ori = 15:15:180;
repeatNum = size(ANN_response_non_ori,2);

%% remove 0 response(have benn discarded)
% non_ori_num = size(ANN_response_non_ori,3);
% effective_neurons = [];
% for neuron = 1:non_ori_num
%     has_zero = any(any(ANN_response_non_ori(:,:,neuron)==0));
%     if ~has_zero
%         effective_neurons = [effective_neurons, neuron];
%     end
% end
% effective_response = ANN_response_non_ori(:,:,effective_neurons);

%% get X and Y
effective_response = ANN_response_non_ori(:,:,:);
X_response = get_X(effective_response);
Y = get_Y(effective_response, ori);

%% predicted
rounds = 1000;

samples = length(ori) * repeatNum;
r_mean_rounds = zeros(rounds,1);
p_mean_rounds = zeros(rounds, 1);
r_out_rounds = zeros(rounds, 10);  
p_out_rounds = zeros(rounds, 10);
MAE_rounds = zeros(rounds, 10);
predicted_all_rounds = zeros(samples, rounds);
Y_test_all_rounds = zeros(samples, rounds);

for i = 1:rounds

    %% plsr 10-fold for all groups
    [~, r_out, p_out, MAE, Weights, predicted_all, Y_test_all] = plsr_10_cv(X_response, Y);
    [r_mean, p_mean] = corr(predicted_all, Y_test_all);
    [r_max, r_max_index] = max(r_out);
    weights_best = Weights(:, r_max_index);
    r_mean_rounds(i) = r_mean;
    p_mean_rounds(i) = p_mean;
    r_out_rounds(i,:) = r_out;
    p_out_rounds(i,:) = p_out;
    MAE_rounds(i,:) = MAE;
    predicted_all_rounds(:,i) = predicted_all;
    Y_test_all_rounds(:,i) = Y_test_all;
        
    disp(fprintf('Round %d prediction ends: r = %0.4f, p=%0.2f', i, r_mean, max(p_out)));
end


r_mean_rounds_median = median(r_mean_rounds);
[~, median_index] = min(abs(r_mean_rounds - r_mean_rounds_median));
all_groups_1000_rounds_median_results.name = net_name;
all_groups_1000_rounds_median_results.r_mean = r_mean_rounds(median_index);
all_groups_1000_rounds_median_results.p_mean = p_mean_rounds(median_index);
all_groups_1000_rounds_median_results.r_out = r_out_rounds(median_index,:);
all_groups_1000_rounds_median_results.p_out = p_out_rounds(median_index,:);
all_groups_1000_rounds_median_results.MAE = MAE_rounds(median_index,:);
all_groups_1000_rounds_median_results.predicted_all =  predicted_all_rounds(:,median_index);
all_groups_1000_rounds_median_results.Y_test_all = Y_test_all_rounds(:, median_index);


disp(fprintf('prediction ends: r = %0.4f, p=%0.2f', r_mean_rounds(median_index), p_mean_rounds(median_index)));


file_name = 'r_mean_rounds.mat';
full_path = fullfile(results_path, file_name);
save(full_path, "r_mean_rounds");

file_name = 'p_mean_rounds.mat';
full_path = fullfile(results_path, file_name);
save(full_path, "p_mean_rounds");

file_name = 'all_groups_1000_rounds_median_results.mat';
full_path = fullfile(results_path, file_name);
save(full_path, "all_groups_1000_rounds_median_results");

delete(poolobj);
