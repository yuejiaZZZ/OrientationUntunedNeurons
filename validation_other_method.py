import os
import glob
from natsort import natsorted
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import random

from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import argparse


ori = np.arange(-45, 121, 15)
neg_ori_index = np.where(ori <= 0)
ori[neg_ori_index] += 180
sf = np.array([0.25, 0.5, 1, 2, 4, 8])
sz = np.array([1, 2, 3])
osiNum = len(sf) * len(sz)  # 18
oln = len(ori)

def loadmat_data(filename):
    file = loadmat(filename)
    name = list(file.keys())
    data = file[name[3]]
    return data

def get_specified_1D_item(all_results, item):

    all_value = []
    for i in range(len(all_results)):
        temp = all_results[i][item]
        all_value.append(temp)
    return all_value

def CV_spilt_dataset(sample_numbers, oriNum, k):
    repeatNum= np.int(sample_numbers/oriNum)
    indices = np.zeros(sample_numbers)
    for ori in range(oriNum):
        j = np.arange(repeatNum*ori, repeatNum*(ori+1), 1)
        random_numbers = np.random.permutation(repeatNum)
        replaced_index = random_numbers >= k
        replaced_number = np.sum(replaced_index)
        random_numbers[replaced_index] = np.random.randint(0, k-1, replaced_number)
        indices[j] = random_numbers
    return indices

def Regress_10_cv(X_response_non_ori, Y, class_num, model):
    k = 10
    indices = CV_spilt_dataset(X_response_non_ori.shape[0], class_num, k)
    r_out = np.zeros(k)
    p_out = np.zeros(k)
    Mae = np.zeros(k)
    Y_predicted_all = np.array([])
    Y_test_all = np.array([])
    for i in range(k):
        test = (indices == i)
        train = ~test
        X_train = X_response_non_ori[train, :]
        Y_train = Y[train]
        X_test = X_response_non_ori[test, :]
        Y_test = Y[test]
        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_test_z = scaler.fit_transform(X_test)
        
        if model == 'LR': 
            reg = LinearRegression()
        if model == 'SVR':
            reg = SVR(kernel='rbf', C=20, gamma='auto')
        if model == 'PLSR':
            reg = PLSRegression(n_components=10)
        if model == 'RFR':
            reg = RandomForestRegressor(n_estimators=100, random_state=0)
        if model == 'Ridge':
            reg = RidgeCV(alphas=[0.1,0.3,0.5,0.7,1])
        if model == 'LASSO':
            Lambdas = np.logspace(-5, 2, 200)
            lasso_cv = LassoCV(alphas=Lambdas, cv=10, max_iter=1000)
            lasso_cv.fit(X_train_z, Y_train)

            lasso_best_alpha = lasso_cv.alpha_  # 0.06294988990221888

            reg = Lasso(alpha=lasso_best_alpha, max_iter=1000)
        if model == 'XGBR':
            reg = XGBRegressor(
                n_estimators=100,
                max_depth=7,
                eta=0.1,
                subsample=0.7,
                colsample_bytree=0.8
            )
                    
        reg.fit(X_train_z, Y_train)
        Y_predicted = np.squeeze(reg.predict(X_test_z))
        r_out[i], p_out[i] = pearsonr(Y_test, Y_predicted)
        Mae[i] = np.mean(abs(Y_test-Y_predicted))
        Y_predicted_all = np.hstack([Y_predicted_all,Y_predicted])
        Y_test_all = np.hstack([Y_test_all, Y_test])
    # r_mean = np.mean(r_out)
    r_mean, p = pearsonr(Y_test_all, Y_predicted_all)
    MAE = np.mean(Mae)
    return r_mean, MAE, Y_predicted_all, Y_test_all


def main():
    parser = argparse.ArgumentParser('Validation of other methods', add_help=False)
    # Dataset parameters
    parser.add_argument('-m', '--monkey_name', 
                        default='MB_CC', 
                        type=str, help='select monkey')
    parser.add_argument('-r', '--regression_method', 
                        default='LR', 
                        type=str, help='select regression_method')
    parser.add_argument('-o', '--outpath',
                        default='/NonOriSelect/val_results',
                        type=str, help='outpath')
    parser.add_argument('-iter', '--iterations',
                    default=1000,
                    type=int, help='outpath')
    
    args = parser.parse_args()
    monkey = args.monkey_name
    regress_model = args.regression_method
    outpath = args.outpath
    os.makedirs(os.path.join(outpath, monkey), exist_ok=True)
    
    
    if monkey == 'MB_CC':
        dir_path = '/MB_CC/Analysis'
    if monkey == 'MC_CC':
        dir_path = '/MC_CC/Analysis'
    if monkey == 'ME':
        dir_path = '/ME/Analysis'
    if monkey == 'MA':
        dir_path = '/MA/Analysis'
    if monkey == 'MD':
        dir_path = '/MD/Analysis'
    
    data_path = os.path.join(dir_path, 'response_data/X_response_all.mat')
    X_response_all = np.squeeze(loadmat_data(data_path))
    data_path = os.path.join(dir_path, 'response_data/Y_all.mat')
    Y_all = np.squeeze(loadmat_data(data_path))
    ori_cell= np.squeeze(loadmat_data(os.path.join(dir_path, 'ori_cell.mat')))
    bool_ori_cell = ori_cell.astype(bool)
    
    group_num = X_response_all.shape[2]
    iterations = args.iterations
    r_matrix = np.zeros((group_num, iterations))
    MAE_matrix = np.zeros((group_num, iterations))
    for the_group in range(group_num):
        X_response = X_response_all[:,:,the_group]
        X_response_non_ori = X_response[:,~bool_ori_cell]
        Y = Y_all[:,the_group]
        
        for iteration in range(iterations):
            r_mean, MAE, Y_predicted_all, Y_test_all = Regress_10_cv(X_response_non_ori, Y, len(ori), model=regress_model)
            # print(f'r_mean = {r_mean}, MAE = {MAE}')
            r_matrix[the_group, iteration] = r_mean
            MAE_matrix[the_group, iteration] = MAE
    

    filename = 'r_matrix_' + regress_model + '.npy'
    file_path = os.path.join(outpath, monkey, filename)
    print(file_path)
    np.save(file_path, r_matrix)
    
    filename = 'MAE_matrix_' + regress_model + '.npy'
    file_path = os.path.join(outpath, monkey, filename)
    print(file_path)
    np.save(file_path, MAE_matrix)
    
    
if __name__ == "__main__":
    main()
    
    
