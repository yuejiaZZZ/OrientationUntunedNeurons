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


%     subplot(2,5,i)
%     scatter(Y_test, predicted, 15, 'filled', 'Marker', 'o');
%     hold on
%     % fitting
%     coefficients = polyfit(Y_test, predicted, 1);
%     xFit = linspace(min(Y_test), max(Y_test), 100);
%     yFit = polyval(coefficients, xFit);
%     plot(xFit, yFit, 'r-')
%     xticks(ori);
%     xtickangle(65);
%     yticks(ori);
%     set(gca, 'FontSize', 6, 'FontAngle', 'italic');
%     xlabel('True Orientation(deg)', 'FontSize', 8, 'FontAngle', 'normal');
%     ylabel('Predicted Orientation(deg)', 'FontSize', 8, 'FontAngle', 'normal');
%     title(sprintf('cv %d', i), 'FontSize', 12, 'FontAngle', 'normal');
%     txt = {sprintf('r = %0.2f', r_out(i)), sprintf('MAE = %0.2f', MAE(i)), sprintf('p = %0.4f', p_out(i))};
%     text( 'string',txt, 'Units','normalized','position',[0.5,0.15],  'FontSize',6,'FontWeight','Bold','FontName','Times New Roman');
end
r_mean = mean(r_out);
Weights = Beta(2:end, :);
end