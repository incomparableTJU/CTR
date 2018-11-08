# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:08:15 2018

@author: weiqing
"""

'''
这次模型把新用户和老用户区别对待
1、先对去除历史信息的全量数据做一次训练
2、对于新用户可以直接提交上次训练的结果
3、对于老用户将上次训练的结果和历史信息组合再一起训练
4、两次训练可以使用同一个模型，但是要使用不同的参数
'''

import pandas as pd
import numpy as np
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle,build_train_dataset,cal_log_loss,submmit_result
from smooth import BayesianSmoothing
import gen_smooth_features as smooth_features
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold,train_test_split
import lightgbm as lgb

params = {
    'max_depth': 8,                 #4
#    'min_data_in_leaf': 40,-
    'feature_fraction': 0.55,       #1
    'learning_rate': 0.04,          #0.04
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'verbose': -1,
    'metric': 'binary_logloss',
#    'max_bin':180,
    'bagging_seed':3,
    'feature_fraction_seed':3,
#    'num_leaves':200
    'lambda_l2':1.5,
    'lambda_l1':1.5
}
def offline(train_data, cv_data):
    
    #剔除历史数据，保留老用户的历史数据
    history_cols = ['user_id_cvr_smooth','user_id_buy_count']
    old_user_data_train = train_data[history_cols]
    old_user_data_test = cv_data[history_cols]
#    train_data.drop(history_cols, axis=1, inplace=True)
#    cv_data.drop(history_cols, axis=1, inplace=True)
    
    #对数据集进行训练
    train_Y = train_data['is_trade']
    cv_Y = cv_data['is_trade']
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    
    kf = KFold(len(train_data), n_folds = 5, shuffle=True, random_state=520)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((cv_data.shape[0], 5))
    for i, (train_index, cv_index) in enumerate(kf):
        
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        print('第{}次训练...'.format(i))
        lgb_train = lgb.Dataset(train_feat.values, train_Y.loc[train_index])
        lgb_cv = lgb.Dataset(cv_feat.values, train_Y.loc[cv_index])
        gbm = lgb.train(params=params,
                        train_set=lgb_train,
                        num_boost_round=6000,
                        valid_sets=lgb_cv,
                        verbose_eval=False,
                        early_stopping_rounds=200)
        #评价特征的重要性
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        
        predict_train = gbm.predict(train_feat.values)
        predict_cv = gbm.predict(cv_feat.values)
        
        test_preds[:,i] = gbm.predict(cv_data.values)
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        print(gbm.best_iteration)
        print(gbm.best_score)
        print('   训练损失:',cal_log_loss(predict_train, train_Y.loc[train_index]))
        print('   测试损失:',cal_log_loss(predict_cv, train_Y.loc[cv_index]))
    test_preds = np.median(test_preds,axis=1)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/4, train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('验证损失:',cal_log_loss(test_preds, cv_Y))
    #划分出新老用户的分数并计算损失情况
    train_old_user_index = old_user_data_train.loc[old_user_data_train.user_id_cvr_smooth!=-1,:].index
    test_old_user_index = old_user_data_test.loc[old_user_data_test.user_id_cvr_smooth!=-1,:].index
    train_new_user_index = old_user_data_train.loc[old_user_data_train.user_id_cvr_smooth==-1,:].index
    test_new_user_index = old_user_data_test.loc[old_user_data_test.user_id_cvr_smooth==-1,:].index
    
    train_old_score = cv_preds[train_old_user_index]
    test_old_score = test_preds[test_old_user_index]
    train_new_score = cv_preds[train_new_user_index]
    test_new_score = test_preds[test_new_user_index]
    
    new_train_data = old_user_data_train.loc[old_user_data_train.user_id_cvr_smooth!=-1,:]
    new_test_data = old_user_data_test.loc[old_user_data_test.user_id_cvr_smooth!=-1,:]
    new_train_data['y'] = train_old_score
    new_test_data['y'] = test_old_score
    new_train_Y = train_Y[train_old_user_index]
    new_test_Y = cv_Y[test_old_user_index]
    #对老用户单独训练
    clf = LogisticRegression(C=12, fit_intercept=True, max_iter=3000,class_weight={0:0.5, 1:0.5})
    clf.fit(X=new_train_data.values, y=new_train_Y)
    
    train_LR_score = clf.predict_proba(new_train_data.values)[:,1]
    test_LR_score = clf.predict_proba(new_test_data.values)[:,1]
    
    cv_preds[train_old_user_index] = train_LR_score
    test_preds[test_old_user_index] = test_LR_score
    #记录老用户的损失情况
    print('LR train:',cal_log_loss(train_LR_score, new_train_Y))
    print('LR test:',cal_log_loss(test_LR_score, new_test_Y))
    #拼接结果看看总体的损失情况
    print('All train:',cal_log_loss(cv_preds, train_Y))
    print('ALL test:',cal_log_loss(test_preds, cv_Y))
    return test_preds, feat_imp

def online(train_data, cv_data, test_data):
    
    #剔除历史数据，保留老用户的历史数据
    cv_data.index += len(train_data)
    train_data = pd.concat([train_data, cv_data],axis=0)
    history_cols = ['user_id_cvr_smooth','user_id_buy_count']
    old_user_data_train = train_data[history_cols]
    old_user_data_test = test_data[history_cols]
    
    #对数据集进行训练
    train_Y = train_data['is_trade']
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
    kf = KFold(len(train_data), n_folds = 5, shuffle=True, random_state=520)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((test_data.shape[0], 5))
    for i, (train_index, cv_index) in enumerate(kf):
        
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        
        print('第{}次训练...'.format(i))
        lgb_train = lgb.Dataset(train_feat.values, train_Y.loc[train_index])
        lgb_cv = lgb.Dataset(cv_feat.values, train_Y.loc[cv_index])
        gbm = lgb.train(params=params,
                        train_set=lgb_train,
                        num_boost_round=6000,
                        valid_sets=lgb_cv,
                        verbose_eval=False,
                        early_stopping_rounds=200)
        #评价特征的重要性
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        
        predict_train = gbm.predict(train_feat.values)
        predict_cv = gbm.predict(cv_feat.values)
        
        test_preds[:,i] = gbm.predict(test_data.values)
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        
        feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)
        print(gbm.best_iteration)
        print(gbm.best_score)
        print('   训练损失:',cal_log_loss(predict_train, train_Y.loc[train_index]))
        print('   测试损失:',cal_log_loss(predict_cv, train_Y.loc[cv_index]))
    test_preds = np.median(test_preds,axis=1)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/4, train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))

    #划分出新老用户的分数并计算损失情况
    train_old_user_index = old_user_data_train.loc[old_user_data_train.user_id_cvr_smooth!=-1,:].index
    test_old_user_index = old_user_data_test.loc[old_user_data_test.user_id_cvr_smooth!=-1,:].index
#    train_new_user_index = old_user_data_train.loc[old_user_data_train.user_id_cvr_smooth==-1,:].index
#    test_new_user_index = old_user_data_test.loc[old_user_data_test.user_id_cvr_smooth==-1,:].index
    
    train_old_score = cv_preds[train_old_user_index]
    test_old_score = test_preds[test_old_user_index]
#    train_new_score = cv_preds[train_new_user_index]
#    test_new_score = test_preds[test_new_user_index]
    
    new_train_data = old_user_data_train.loc[old_user_data_train.user_id_cvr_smooth!=-1,:]
    new_test_data = old_user_data_test.loc[old_user_data_test.user_id_cvr_smooth!=-1,:]
    new_train_data['y'] = train_old_score
    new_test_data['y'] = test_old_score
    new_train_Y = train_Y[train_old_user_index]

    #对老用户单独训练
    clf = LogisticRegression(C=12, fit_intercept=True, max_iter=3000,class_weight={0:0.5, 1:0.5})
    clf.fit(X=new_train_data.values, y=new_train_Y)
    
    train_LR_score = clf.predict_proba(new_train_data.values)[:,1]
    test_LR_score = clf.predict_proba(new_test_data.values)[:,1]
    
    cv_preds[train_old_user_index] = train_LR_score
    test_preds[test_old_user_index] = test_LR_score
    #记录老用户的损失情况
    print('LR train:',cal_log_loss(train_LR_score, new_train_Y))
    #拼接结果看看总体的损失情况
    print('All train:',cal_log_loss(cv_preds, train_Y))
    
    submmit_result(test_preds,'old_and_new')
    return test_preds, feat_imp

if __name__ == '__main__':
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data = load_pickle(path=cache_pkl_path +'test_data')
    
    
    cols = ['user_gender_id','user_age_level','user_occupation_id','user_star_level',\
            'item_brand_id','item_city_id','query_item_second_cate_sim','query_item_second_cate_sim',\
            'item_id_buy_count','item_brand_id_buy_count','shop_id_buy_count',\
            'item_id_cvr_smooth','item_brand_id_cvr_smooth','shop_id_cvr_smooth',\
            'max_cp_cvr','min_cp_cvr','mean_cp_cvr']
    for i in cols:
        train_data[i].replace(to_replace=-1,value=np.nan,inplace=True)
        cv_data[i].replace(to_replace=-1,value=np.nan,inplace=True)
        test_data[i].replace(to_replace=-1,value=np.nan,inplace=True)
    
    online(train_data, cv_data, test_data)
#    predict_test, feat_imp = offline(train_data, cv_data)
    t1 = time.time()
    print('训练用时:',t1-t0)