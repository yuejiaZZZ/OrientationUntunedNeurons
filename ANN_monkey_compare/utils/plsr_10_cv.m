function [r_mean, r_out, p_out, MAE, Weights, predicted_all, Y_test_all] = plsr_10_cv(X_response, Y)

ori = 15:15:180;
Num_AllComp = 10;
% indices = crossvalind('Kfold', size(X_response, 1), 10);
indices = CV_split_dataset(size(X_response, 1), length(ori), 10);
best_ncomp_10fold = zeros(1,10);
r_out = zeros(1,10);
p_out = zeros(1,10);
MAE = zeros(1,10);
predicted_all = [];
Y_test_all = [];
Beta = [];

% figure;
for i = 1:10
    test = (indices == i); train = ~test;
    X_train = X_response(train,:);
    Y_train = Y(train);
    X_test = X_response(test,:);
    Y_test = Y(test);
    [X_train_z,meanx,devx] = zscore(X_train);
    [Y_train_z,meany,devy] = zscore(Y_train);
    
    % the inner 10-fold, to find the best ncomp
    [~,~,~,~,~,~,mse_t] = plsregress(X_train_z, Y_train, Num_AllComp,'CV',10);
    [~,idx_minmse] = min(mse_t(2,:));
    ncomp_best = idx_minmse;
    best_ncomp_10fold(i) = ncomp_best;
    
    % out
    [~,~,~,~,beta] = plsregress(X_train_z, Y_train, ncomp_best);
    X_test_z = zscore(X_test);
    test_features = [ones(size(X_test_z, 1), 1), X_test_z];
    predicted = test_features * beta;
    [r_out(i), p_out(i)] = corr(Y_test, predicted);
    MAE(i) = mean(abs(Y_test - predicted));
    predicted_all = [predicted_all;predicted];
    Y_test_all = [Y_test_all;Y_test]; 
    Beta = [Beta, beta];


end
r_mean = mean(r_out);
Weights = Beta(2:end, :);
end