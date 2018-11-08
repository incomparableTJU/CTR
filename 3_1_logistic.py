# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:40:38 2018

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
from sklearn.cross_validation import KFold,train_test_split
import matplotlib.pyplot as plt

rate = 0.75
def LR_offline(train_data, cv_data):
    train_Y = train_data['is_trade']
    cv_Y = cv_data['is_trade']
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    
    fold = 5
    kf = KFold(len(train_data), n_folds = fold, shuffle=True, random_state=520)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((cv_data.shape[0], fold))
    for i, (train_index, cv_index) in enumerate(kf):
        
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        clf = LogisticRegression(C=1.2, fit_intercept=True, max_iter=3000,class_weight={0:0.5, 1:0.5})
        clf.fit(X=train_feat.values, y=train_Y[train_index])
        
        predict_train = clf.predict_proba(train_feat.values)[:,1]
        predict_cv = clf.predict_proba(cv_feat.values)[:,1]
        predict_test = clf.predict_proba(cv_data.values)[:,1]
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
    predict_test = np.median(test_preds,axis=1)
    print('验证损失:',cal_log_loss(predict_test, cv_Y))
    
def LR_online(train_data, cv_data, test_data):
    train_data = pd.concat([train_data, cv_data],axis=0)
    train_data = build_train_dataset(train_data, rate)
    train_data.reset_index(inplace=True,drop=True)
    train_Y = train_data['is_trade']
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
    fold = 5
    kf = KFold(len(train_data), n_folds = fold, shuffle=True, random_state=520)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((test_data.shape[0], fold))
    for i, (train_index, cv_index) in enumerate(kf):
        
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        clf = LogisticRegression(C=1.2, fit_intercept=True, max_iter=3000,class_weight={0:0.5, 1:0.5})
        clf.fit(X=train_feat.values, y=train_Y[train_index])
        
        predict_train = clf.predict_proba(train_feat.values)[:,1]
        predict_cv = clf.predict_proba(cv_feat.values)[:,1]
        predict_test = clf.predict_proba(test_data.values)[:,1]
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print('训练损失:',cal_log_loss(train_preds/4, train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    submmit_result(predict_test, 'LR')
if __name__ == '__main__':
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data_onehot')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data_onehot')
    test_data = load_pickle(path=cache_pkl_path +'test_data_onehot')
    
    drop_cols = ['user_id','item_id']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
    
    print('train shap:',train_data.shape)
    print('cv shape', cv_data.shape)
    print('test shape', test_data.shape)
    
#    LR_offline(train_data, cv_data)
    LR_online(train_data, cv_data, test_data)