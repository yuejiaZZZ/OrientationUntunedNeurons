from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from scipy.stats import f_oneway, mannwhitneyu, ttest_ind, shapiro, levene, kstest, kruskal
import numpy as np
from statsmodels.stats.multitest import multipletests

def loadmat_data(filename):
    file = loadmat(filename)
    name = list(file.keys())
    data = file[name[3]]
    return data

def normlized_tuning(neuron_response,mode:str):
    if mode == 'zscore':
        scaler = StandardScaler().fit(neuron_response)
        neuron_reponse_normlized = scaler.transform(neuron_response)
    elif mode == 'maxmin':
        scaler = MinMaxScaler()
        neuron_reponse_normlized = scaler.fit_transform(neuron_response)
    elif mode == 'all':    
        scaler = StandardScaler().fit(neuron_response)
        neuron_reponse_zscore = scaler.transform(neuron_response)
        scaler = MinMaxScaler()
        neuron_reponse_normlized = scaler.fit_transform(neuron_reponse_zscore)
    else:
        return None
    return neuron_reponse_normlized


def get_specified_1D_item(all_results, item):
    # i 代表第几组（横坐标），item代表哪一个指标（纵坐标）
    all_value = []
    for i in range(len(all_results)):
        temp = all_results[i][item]
        all_value.append(temp)
    return all_value


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.01:
        return "*"
    return "ns"

def convert_pvalue_to_str(pvalue):

    return f'p = {pvalue:.3f}'


def load_ori_cell_index_path(Monkey_path, cell_pattern:str):
    if cell_pattern=='target':
        Monkey_target_path = os.path.join(os.path.dirname(Monkey_path), 'Analysis_learn_form_yu')
        data_path = os.path.join(Monkey_target_path, 'targetCell.mat')
    elif cell_pattern=='ori':
        data_path = os.path.join(Monkey_path, 'ori_cell_index.mat')
    return data_path


def difference_compare_two_group(group1, group2):
    # 检验正态性
    _, p_normal_group1 = shapiro(group1)
    _, p_normal_group2 = shapiro(group2)
    _, p_normal_group1 = kstest(group1, 'norm')
    _, p_normal_group2 = kstest(group2, 'norm')
    print(p_normal_group1, p_normal_group2)
    # 检验方差齐性
    _, p_levene = levene(group1, group2)

    # 设置显著性水平
    alpha = 0.05

    # 判断是否进行 t 检验（正态且方差齐）或 Mann-Whitney U 检验（不符合要求）
    if p_normal_group1 > alpha and p_normal_group2 > alpha and p_levene> alpha:
        # 使用 t 检验
        t_statistic, p_value = ttest_ind(group1, group2)
        test_used = 't-test'
        n = len(group1) + len(group2)
        z_statistic = t_statistic / (n ** 0.5)
    else:
        u_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        test_used = 'Mann-Whitney U'
        
        # 计算 z 值的近似值
        n1 = len(group1)
        n2 = len(group2)
        mu_u = n1 * n2 / 2
        sigma_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
        z_statistic = (u_statistic - mu_u) / sigma_u

    # 打印结果
    print(f'Test used: {test_used}, statistic: {z_statistic}, \n p-value: {p_value:.3e} \n p-value: {p_value:.4f} ')
    return p_value


def t_to_z(group1, group2, t_statistic):
    n = len(group1) + len(group2)
    z_statistic = t_statistic / (n ** 0.5)
    return z_statistic


def u_to_z(group1, group2, u_statistic):
    # 计算 z 值的近似值
    n1 = len(group1)
    n2 = len(group2)
    mu_u = n1 * n2 / 2
    sigma_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    z_statistic = (u_statistic - mu_u) / sigma_u
    return z_statistic


def find_nth_number(arr, n, number:int):
    count = -1
    for index, value in enumerate(arr):
        if value == number:
            count += 1
            if count == n:
                return index
            
            
def compare_r(r_1,r_2,r_3):
    # 正态性检验
    _, p_normal_group0 = shapiro(r_1)
    _, p_normal_group1 = shapiro(r_2)
    _, p_normal_group2 = shapiro(r_3)
    
    # 方差齐性检验
    _, p_levene = levene(r_1, r_2, r_3)
    
    alpha = 0.5
    # 定义两两组合
    pairs = [(r_1, r_2),
            (r_2, r_3),
            (r_3, r_1)]
    if all(p > alpha for p in [p_normal_group0, p_normal_group1, p_normal_group2]) and p_levene > alpha:
        print('参数, Anova(F) and t-test(z)')
        statistic_anova, p_anova = f_oneway(r_1, r_2, r_3)
        statistic_test = [ttest_ind(pair[0], pair[1])[0] for pair in pairs]
        z_statistic = [t_to_z(pair[0], pair[1], statistic_test[i]) for i,pair in enumerate(pairs)]
        p_values_test = [ttest_ind(pair[0], pair[1])[1] for pair in pairs]
        reject, p_values_fdr, _, _ = multipletests(p_values_test, method='fdr_bh')
    else:
        print('非参数, Kruskal-Wallis test(H) and Mann-Whitney U test(z)')
        statistic_anova, p_anova = kruskal(r_1, r_2, r_3)
        statistic_test = [mannwhitneyu(pair[0], pair[1])[0] for pair in pairs]
        z_statistic = [u_to_z(pair[0], pair[1], statistic_test[i]) for i,pair in enumerate(pairs)]
        p_values_test = [mannwhitneyu(pair[0], pair[1])[1] for pair in pairs]
        reject, p_values_fdr, _, _ = multipletests(p_values_test, method='fdr_bh')
    print(f'mutil-group statistic: {statistic_anova:.3f}')
    print(f'mutil-group p-value: {p_anova:.3e},  p-value: {p_anova:.4f} ')
    print(f'pairs reject: {reject}')
    print(f'pairs z-value: {z_statistic}')
    print(f'pairs p-value: {p_values_fdr}')
    return p_values_fdr