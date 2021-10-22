# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:07:21 2021

@author: zongsing.huang
"""

import itertools

import numpy as np

#%% 題庫
benchmark = np.array([[ 0, 19, 92, 29, 49, 78,  6],
                      [19,  0, 21, 85, 45, 16, 26],
                      [92, 21,  0, 24, 26, 87, 47],
                      [29, 85, 24,  0, 76, 17,  8],
                      [49, 45, 26, 76,  0, 90, 27],
                      [78, 16, 87, 17, 90,  0, 55],
                      [ 6, 26, 47,  8, 27, 55,  0]])

#%% 函數定義
def fitness(X, benchmark):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    D = X.shape[1]
    F = np.zeros(P)
    
    for i in range(P):
        X_new = np.append(X[i], X[i, 0])
        
        for j in range(D):
            st = X_new[j].astype(int)
            ed = X_new[j+1].astype(int)
            F[i] += benchmark[st, ed]
    
    return F

def swap(X):
    D = X.shape[0]
    idx = np.arange(D)
    comb = list(itertools.combinations(idx, 2))
    X_new = np.zeros([len(comb), D])
    
    for i, (j, k) in enumerate(comb):
        X_new[i] = X.copy()
        X_new[i, j], X_new[i, k] = X_new[i, k], X_new[i, j]
    
    return X_new

#%% 參數設定
D = benchmark.shaep[1] # 維度
G = 5 # 迭代次數
L = 3 # 禁忌表長度


#%% 初始化
X = np.random.choice(D, size=D, replace=False)
F = fitness(X, benchmark) # 當前最佳適應值
X_gbest = X.copy() # 全域最佳解
F_gbest = F.copy() # 全域最佳適應解
tabu_list = np.zeros([L, D]) # 禁忌表

for g in range(G):
    # 更新
    X_set = swap(X)
    
    # 移除已訪問過的路徑
    P = X_set.shape[0]
    mask1 = np.arange(P)
    mask2 = np.ones(P, dtype=bool)
    for i in range(P):
        for j in range(L):
            if np.array_equal(X_set[i], tabu_list[j]):
                mask2[i] = False
                break
    mask1 = mask1[mask2]
    X_set = X_set[mask1]
    
    # 計算適應值
    F_set = fitness(X_set, benchmark)
    idx = F_set.argmin()
    
    # 適應值更新
    if F_set.min()<F_gbest:
        X_gbest = X_set[idx]
        F_gbest = F_set[idx]
    
    X = X_set[idx].copy()
    
    # 禁止訪問表更新
    tabu_list = np.insert(tabu_list, 0, X, axis=0)
    tabu_list = np.delete(tabu_list, -1, 0)
