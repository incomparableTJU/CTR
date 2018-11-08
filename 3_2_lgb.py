# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:51:11 2018

@author: weiqing
"""
import pandas as pd
import numpy as np
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle,build_train_dataset,cal_log_loss,submmit_result
from smooth import BayesianSmoothing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold,train_test_split
import lightgbm as lgb


params = {
    'max_depth': 6,                 #4
#    'min_data_in_leaf': 40,-
    'feature_fraction': 0.7,       #0.55
    'learning_rate': 0.1,          #0.04
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'verbose': -1,
    'metric': 'binary_logloss',
#    'max_bin':240,
    'bagging_seed':3,
    'feature_fraction_seed':3,
#    'num_leaves':64
#    'lambda_l2': 0.02
#    'lambda_l1':0.05
}
rate = 1
def lgb_online(train_data, cv_data, test_data):
    
    train_data = pd.concat([train_data, cv_data],axis=0)
    train_data = build_train_dataset(train_data, rate)
    train_data.reset_index(inplace=True,drop=True)
    train_Y = train_data['is_trade']
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
    kf = KFold(len(train_data), n_folds = 5, shuffle=True, random_state=520)
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((test_data.shape[0], 5))
    for i, (train_index, cv_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_feat = train_data.loc[train_index]
        cv_feat = train_data.loc[cv_index]
        lgb_train = lgb.Dataset(train_feat.values, train_Y.loc[train_index])
        lgb_cv = lgb.Dataset(cv_feat.values, train_Y.loc[cv_index])
        gbm = lgb.train(params=params,
                        train_set=lgb_train,
                        num_boost_round=6000,
                        valid_sets=lgb_cv,
                        verbose_eval=False,
                        early_stopping_rounds=500)
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
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/4, train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    submmit_result(predict_test, 'LGB')
    return gbm, feat_imp

def lgb_offline(train_data, cv_data):
    
    train_data = build_train_dataset(train_data, rate)
    train_data.reset_index(inplace=True,drop=True)
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
                        early_stopping_rounds=100)
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
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print(params)
    print('训练损失:',cal_log_loss(train_preds/4, train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('验证损失:',cal_log_loss(predict_test, cv_Y))
    
    return gbm,feat_imp
if __name__ == '__main__':
    
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data = load_pickle(path=cache_pkl_path +'test_data')
    
#    train_data.drop(train_data.index[train_data.day==train_data.day.min()],inplace=True, axis=0)
#    cols = ['user_gender_id','user_age_level','user_occupation_id','user_star_level',\
#            'item_brand_id','item_city_id','query_item_second_cate_sim','query_item_second_cate_sim',\
#            'user_id_buy_count','item_id_buy_count','item_brand_id_buy_count','shop_id_buy_count',\
#            'user_id_cvr_smooth','item_id_cvr_smooth','item_brand_id_cvr_smooth','shop_id_cvr_smooth',\
#            'max_cp_cvr','min_cp_cvr','mean_cp_cvr']
#    for i in cols:
    train_data.replace(to_replace=-1,value=np.nan,inplace=True)
    cv_data.replace(to_replace=-1,value=np.nan,inplace=True)
    test_data.replace(to_replace=-1,value=np.nan,inplace=True)
        
    drop_cols = ['user_id_1day_cvr','item_id_visit#30M/1H', 'shop_id_visit#24H','item_id_visit#4H','user_id_visit#30M',\
                 'user_id_buy_count','day','user_hour_shop_search','item_id_visit#30M','item_id','user_id_cvr_smooth', 'shop_id_cvr_smooth','item_id_cvr_smooth','item_brand_id_cvr_smooth']
    train_data.drop(drop_cols, inplace=True, axis=1)
    cv_data.drop(drop_cols, inplace=True, axis=1)
    test_data.drop(drop_cols, inplace=True, axis=1)
#    gbm, feat_imp = lgb_online(train_data, cv_data,test_data)
    gbm, feat_imp = lgb_offline(train_data, cv_data)
    t1 = time.time()
    print('训练时间:',t1-t0)
    