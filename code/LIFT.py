# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:16:24 2018

@author: FNo0
"""

import pandas as pd
import numpy as np
import scipy.io as sio
import math
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import xgboost as xgb
import scipy

def load_data(mat_path):
    # 矩阵文件
    data = sio.loadmat(mat_path)
    # 解析
    train_data = data['train_data']
    train_target = data['train_target']
    test_data = data['test_data']
    test_target = data['test_target']
    # 返回
    return train_data,train_target,test_data,test_target

def find_key_instances(train_data,train_target,ratio):
    num_train,dim = np.shape(train_data)
    num_class = len(train_target)
    P_Centers = []
    N_Centers = []
    # Find key instances of each label
    for i in range(num_class):
        print('Performing clustering for the ' + str(i+1) + '/' + str(num_class) + '-th class')
        p_idx = list(np.where(train_target[i] == 1)[0]) # 第i个标签为1的训练集编号
        n_idx = list(np.where(train_target[i] == -1)[0]) # 第i个标签为-1的训练集编号
        p_data = train_data[p_idx,:] # 第i个标签为1的训练集
        n_data = train_data[n_idx,:] # 第i个标签为-1的训练集
        k1 = min(math.ceil(len(p_idx)*ratio),math.ceil(len(n_idx)*ratio)) # 进一法
        k2 = k1
        if k1 == 0: # 全为正例或者全为负例
            POS_C = []
            zero_kmeans = KMeans(n_clusters = min(50, num_train),random_state = 0).fit(train_data)
            NEG_C = zero_kmeans.cluster_centers_
        else:
            if len(p_data) == 1: # 只有一个正例,不用再聚类
                POS_C = p_data.copy()
            else:
                p_kmeans = KMeans(n_clusters = k1,random_state = 0).fit(p_data)
                POS_C = p_kmeans.cluster_centers_
            if len(n_data) == 1: # 只有一个负例,不用再聚类
                NEG_C = n_data.copy()
            else:
                n_kmeans = KMeans(n_clusters = k2,random_state = 0).fit(n_data)
                NEG_C = n_kmeans.cluster_centers_
        P_Centers.append(POS_C) # 每个类别下正例的中心点
        N_Centers.append(NEG_C) # 每个类别下负例的中心点
    # return
    return P_Centers,N_Centers

def train(train_data,train_target,P_Centers,N_Centers):
    num_train,dim = np.shape(train_data)
    num_class = len(train_target)
    Models = []
    # Perform representation transformation and training
    for i in range(num_class):
        print('Building classiers : ' + str(i+1) + '/' + str(num_class))
        centers = np.vstack((P_Centers[i],N_Centers[i]))
        num_center = len(centers)
        data = np.empty(shape = [0,num_center]) # 一定要使用np.empty,不然data = np.vstack(())会出错
        if num_center >= 5000:
            print('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...')
            break
        else:
            # 分块求
            blocksize = 5000 - num_center
            num_block = math.ceil(num_train/blocksize)
            # 块数为1时,此for循环无作用
            for j in range(1,num_block):
                low = (j - 1) * blocksize + 1
                high = j * blocksize
                tmp_mat = np.vstack((centers,train_data[low-1:high,:]))
                Y = scipy.spatial.distance.pdist(tmp_mat,'euclidean')
                Z = scipy.spatial.distance.squareform(Y)
                data = np.vstack((data,Z[num_center:num_center+blocksize,0:num_center]))
            low = (num_block - 1) * blocksize + 1
            high = num_train
            tmp_mat = np.vstack((centers,train_data[low-1:high,:]))
            Y = scipy.spatial.distance.pdist(tmp_mat,'euclidean')
            Z = scipy.spatial.distance.squareform(Y)
            data = np.vstack((data,Z[num_center:num_center+high-low+1,0:num_center]))
        # 特征转换后的训练集特征及其对应的标签    
        training_instance_matrix = np.hstack((train_data,data))
        training_label_vector = train_target[i]
        training_label_vector = np.array([1 if label == 1 else 0 for label in training_label_vector]) # 标签-1转0
        
        # 模型训练
        # subject
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric' : 'error',
                  'eta': 0.03,
                  'max_depth': 6,  # 4 3
                  'colsample_bytree': 0.9,#0.8
                  'subsample': 0.9,
                  'scale_pos_weight': 1,
                  'min_child_weight': 0,  # 2 3
                  'lambda' : 5,
                  'gamma' : 0.1
                  }
        rounds = 240
        dtrain = xgb.DMatrix(training_instance_matrix, label = training_label_vector)
#        watchlist = [(dtrain, 'train')]
#        model = xgb.train(params, dtrain, num_boost_round = rounds,evals = watchlist)
        model = xgb.train(params, dtrain, num_boost_round = rounds)
        
#        model = xgb.XGBClassifier(objective = 'binary:logistic',
#                                  learning_rate = 0.03,
#                                  max_depth = 6,
#                                  colsample_bytree = 0.8,
#                                  subsample = 0.8,
#                                  scale_pos_weight = 1,
#                                  min_child_weight = 0,
#                                  n_estimators = 240)
#        model.booster = 'gbliner'
#        model.fit(training_instance_matrix,training_label_vector)
        
        Models.append(model)
    # return
    return Models

def test(train_data,train_target,test_data,P_Centers,N_Centers,Models):
    num_test,dim = np.shape(test_data)
    num_class = len(train_target)
    Pre_Labels = pd.DataFrame()
    # Perform representation transformation and testing
    for i in range(num_class):
        centers = np.vstack((P_Centers[i],N_Centers[i]))
        num_center = len(centers)
        data = np.empty(shape = [0,num_center]) # 一定要使用np.empty,不然data = np.vstack(())会出错
        if num_center >= 5000:
            print('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...')
            break
        else:
            # 分块求
            blocksize = 5000 - num_center
            num_block = math.ceil(num_test/blocksize)
            # 块数为1时,此for循环无作用
            for j in range(1,num_block):
                low = (j - 1) * blocksize + 1
                high = j * blocksize
                tmp_mat = np.vstack((centers,test_data[low-1:high,:]))
                Y = scipy.spatial.distance.pdist(tmp_mat,'euclidean')
                Z = scipy.spatial.distance.squareform(Y)
                data = np.vstack((data,Z[num_center:num_center+blocksize,0:num_center]))
            low = (num_block - 1) * blocksize + 1
            high = num_test
            tmp_mat = np.vstack((centers,test_data[low-1:high,:]))
            Y = scipy.spatial.distance.pdist(tmp_mat,'euclidean')
            Z = scipy.spatial.distance.squareform(Y)
            data = np.vstack((data,Z[num_center:num_center+high-low+1,0:num_center]))
        # 特征转换后的测试集特征   
        testing_instance_matrix = np.hstack((test_data,data))
        # 预测
        dtest = xgb.DMatrix(testing_instance_matrix)
        predicted_label = Models[i].predict(dtest)
        
#        predicted_label = Models[i].predict_proba(testing_instance_matrix) 
        
        predicted_label = pd.DataFrame(predicted_label,columns = [str(i)])
        Pre_Labels = pd.concat([Pre_Labels,predicted_label],axis = 1)
    # return
    return Pre_Labels


    
    
    