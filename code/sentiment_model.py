# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:48:06 2018

@author: FNo0
"""

import pandas as pd
import xgboost as xgb
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

def model_xgb(train,test,label,params,rounds):
    train_y = train[label]
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    train_x = train.drop(['content_id','content','sentiment_value','sentiment_word','subject','words'],axis=1)
    test_x = test.drop(['content_id','content','sentiment_value','sentiment_word','subject','words'],axis=1)
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 训练
#    watchlist = [(dtrain, 'train')]
#    bst = xgb.train(params, dtrain, num_boost_round=rounds,evals=watchlist)
    bst = xgb.train(params, dtrain, num_boost_round=rounds)
    # 预测
    predict = bst.predict(dtest)
    predict = pd.DataFrame(predict)
    test_xy = test[['content_id','content']]
    test_xy = pd.concat([test_xy,predict],axis = 1)
    for col in range(0,params['num_class']):
        test_xy.rename(columns = {col : le.inverse_transform(int(col))},inplace = True)
    test_xy.set_index(['content_id','content'],inplace = True)
    test_xy = test_xy.stack()
    test_xy = pd.DataFrame(test_xy)
    test_xy.reset_index(inplace = True)
    test_xy[label] = test_xy['level_2']
    test_xy.drop(['level_2'],axis = 1,inplace = True)
    test_xy.sort_values(0,ascending = False,inplace = True)
    test_xy.drop_duplicates(['content_id','content',label],keep = 'first',inplace = True)
    return test_xy

def get_params_rounds(train):
    # 价格(1273)
    params1 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds1 = 400 
    # 内饰(536)
    params2 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds2 = 400 
    # 动力(2732)
    params3 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds3 = 400 
    # 外观(489)
    params4 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds4 = 400 
    # 安全性(573)
    params5 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds5 = 400 
    # 操控(1036)
    params6 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds6 = 400 
    # 油耗(1082)
    params7 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds7 = 400 
    # 空间(442)
    params8 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds8 = 400 
    # 舒适性(931)
    params9 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds9 = 400 
    # 配置(853)
    params10 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds10 = 400 
    # 整合
    params = [params1,params2,params3,params4,params5,params6,params7,params8,params9,params10]
    rounds = [rounds1,rounds2,rounds3,rounds4,rounds5,rounds6,rounds7,rounds8,rounds9,rounds10]
    # 返回
    return params,rounds