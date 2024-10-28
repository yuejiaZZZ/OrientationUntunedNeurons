function [predicted, r_out, p_out, beta] = plsr(X_train, Y_train, X_test, Y_test, Num_AllComp)


    [X_train_z,meanx,devx] = zscore(X_train);
    [Y_train_z,meany,devy] = zscore(Y_train);
    
    % the inner 10-fold, to find the best ncomp
    [~,~,~,~,~,~,mse_t] = plsregress(X_train_z, Y_train, Num_AllComp,'CV',10);
    [~,idx_minmse] = min(mse_t(2,:));
    ncomp_best = idx_minmse;
    
    % out
    [~,~,~,~,beta] = plsregress(X_train_z, Y_train, ncomp_best);
    X_test_z = zscore(X_test);
    test_features = [ones(size(X_test_z, 1), 1), X_test_z];
    predicted = test_features * beta;
    [r_out, p_out] = corr(Y_test, predicted);
end

