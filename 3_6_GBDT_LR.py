# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:53:53 2018

@author: weiqing
"""

import pandas as pd
import numpy as np
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle,build_train_dataset,cal_log_loss,submmit_result
from smooth import BayesianSmoothing
import gen_smooth_features as smooth_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
import xgboost as xgb
import operator 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder  
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split

# In[]:生成GBDT的特征
def gen_gbdt_feature(clf, train_data,cv_data,test_data):
    
    x = clf.apply(train_data)
    x = x.reshape((x.shape[0],x.shape[1]))
    enc = OneHotEncoder()
    enc.fit(x)
    new_feature_train=enc.transform(x)
    new_feature_train=new_feature_train.toarray()
#    new_feature_train=np.concatenate([train_data.values, new_feature_train],axis=1)
    
    x = gbc.apply(cv_data)
    x = x.reshape((x.shape[0],x.shape[1]))
    new_feature_cv=enc.transform(x)
    new_feature_cv=new_feature_cv.toarray()
#    new_feature_cv=np.concatenate([cv_data.values, new_feature_cv],axis=1)
    
    x = gbc.apply(test_data)
    x = x.reshape((x.shape[0],x.shape[1]))
    new_feature_test=enc.transform(x)
    new_feature_test=new_feature_test.toarray()
#    new_feature_test=np.concatenate([test_data.values, new_feature_test],axis=1)
    
    return new_feature_train,new_feature_cv,new_feature_test

# In[]:
if __name__ == '__main__':
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data_onehot')
    train_Y = load_pickle(path=cache_pkl_path +'train_Y')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data_onehot')
    cv_Y = load_pickle(path=cache_pkl_path +'cv_Y')
    
    test_data = load_pickle(path=cache_pkl_path +'test_data_onehot')
    
    drop_cols = ['user_id','shop_id','item_id','item_brand_id']
    train_id_df = train_data[drop_cols]
    cv_id_df = cv_data[drop_cols]
    test_id_df = test_data[drop_cols]
    drop_cols = ['user_id','shop_id','item_id','item_brand_id','is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
#    train_data, _, train_Y, _ = train_test_split(train_data,
#                                                 train_Y,
#                                                 test_size=0.5)
    
#    gbc = GradientBoostingClassifier(n_estimators=27,learning_rate=0.1,max_depth=6,max_leaf_nodes=35)
    gbc = GradientBoostingClassifier(n_estimators=27,learning_rate=0.1,max_depth=6, max_leaf_nodes=35)
    gbc.fit(train_data.values, train_Y)
    predict_train = gbc.predict_proba(train_data.values)[:,1]
    predict_cv = gbc.predict_proba(cv_data.values)[:,1]
    predict_test = gbc.predict_proba(test_data.values)[:,1]
    
#    print(gbc.get_params)
    print('训练损失:',cal_log_loss(predict_train, train_Y))
    print('测试损失:',cal_log_loss(predict_cv, cv_Y))
    t1 = time.time()
    print('训练用时:',t1-t0)
    
    new_train, new_cv, new_test = gen_gbdt_feature(gbc, train_data, cv_data, test_data)
    print('train shap:',new_train.shape)
    print('cv shape', new_cv.shape)
    print('test shape', new_test.shape)
    
#    LR预测
    clf = LogisticRegression(C=0.8, fit_intercept=True, max_iter=3000,class_weight={0:0.5, 1:0.5})
    clf.fit(X=new_train, y=np.squeeze(train_Y))
    
    predict_train = clf.predict_proba(new_train)[:,1]
    predict_cv = clf.predict_proba(new_cv)[:,1]
    predict_test = clf.predict_proba(new_test)[:,1]
    
    print('训练损失:',cal_log_loss(predict_train, train_Y))
    print('测试损失:',cal_log_loss(predict_cv, cv_Y))
    t1 = time.time()
    print('训练用时:',t1-t0)
    
    submmit_result(predict_test,'GBDT_LR')