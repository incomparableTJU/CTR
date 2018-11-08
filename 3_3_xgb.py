# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:54:37 2018

@author: weiqing
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
import operator
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle,build_train_dataset,cal_log_loss,submmit_result
from smooth import BayesianSmoothing
import gen_smooth_features as smooth_features
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold,train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt


def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1
#        print(i)
    outfile.close() 
    
params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':6,
    'colsample_bytree': 0.85,
    'nthread':4,
#    'gamma':0.6,
    'lambda':1,
    'eta':0.4,
#    'silent':0,
#    'alpha':0.01,
#    'subsample':1,
}

n_round=500

rate = 1
def xgb_offline(train_data, cv_data):
    
    train_data = build_train_dataset(train_data, rate)       
    train_data.reset_index(inplace=True,drop=True)
    train_Y = train_data['is_trade'].values
    cv_Y = cv_data['is_trade'].values
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    print('train shap:',train_data.shape)
    print('cv shape', cv_data.shape)
    
    kf = KFold(len(train_data), n_folds = 5, shuffle=True, random_state=520)
    
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((cv_data.shape[0], 5))
    
    for i, (train_index, cv_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_feat = train_data.iloc[train_index]
        cv_feat = train_data.iloc[cv_index]
        
        train_feat = xgb.DMatrix(train_feat.values, label=train_Y[train_index])
        cv_feat = xgb.DMatrix(cv_feat.values,label=train_Y[cv_index])
        test_feat = xgb.DMatrix(cv_data.values)
        watchlist = [(train_feat, 'train'),(cv_feat, 'val')]

        
        clf = xgb.train(params=params, dtrain=train_feat,num_boost_round=n_round,\
            evals=watchlist,early_stopping_rounds=7,verbose_eval=False)
    
        predict_train = clf.predict(train_feat)
        predict_cv = clf.predict(cv_feat)
        predict_test = clf.predict(test_feat)
        
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        
        print(clf.best_iteration)
        print(clf.best_score)
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
    #特征重要度
    features = train_data.columns
    ceate_feature_map(features)
    importance = clf.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    print('训练损失:',cal_log_loss(train_preds/4, train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    print('验证损失:',cal_log_loss(predict_test, cv_Y))
    return df, clf

def xgb_online(train_data, cv_data, test_data):
    train_data = build_train_dataset(train_data, rate)
    train_data = pd.concat([train_data,cv_data],axis=0)
    
    train_data.reset_index(inplace=True,drop=True)
    train_Y = train_data['is_trade'].values
    
    drop_cols = ['is_trade']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    print('train shap:',train_data.shape)
    
    kf = KFold(len(train_data), n_folds = 5, shuffle=True, random_state=520)
    
    train_preds = np.zeros(train_data.shape[0])
    cv_preds = np.zeros(train_data.shape[0])
    test_preds = np.zeros((test_data.shape[0], 5))
    for i, (train_index, cv_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_feat = train_data.iloc[train_index]
        cv_feat = train_data.iloc[cv_index]
        
        train_feat = xgb.DMatrix(train_feat.values, label=train_Y[train_index])
        cv_feat = xgb.DMatrix(cv_feat.values,label=train_Y[cv_index])
        test_feat = xgb.DMatrix(test_data.values)
        
        watchlist = [(train_feat, 'train'),(cv_feat, 'val')]
        clf = xgb.train(params=params, dtrain=train_feat,num_boost_round=n_round,\
            evals=watchlist,early_stopping_rounds=30,verbose_eval=False)
    
        predict_train = clf.predict(train_feat)
        predict_cv = clf.predict(cv_feat)
        predict_test = clf.predict(test_feat)
        
        train_preds[train_index] += predict_train
        cv_preds[cv_index] += predict_cv
        test_preds[:,i] = predict_test
        
        print(clf.best_iteration)
        print(clf.best_score)
        print('  训练损失:',cal_log_loss(predict_train, train_Y[train_index]))
        print('  测试损失:',cal_log_loss(predict_cv, train_Y[cv_index]))
        
    #特征重要度
    features = train_data.columns
    ceate_feature_map(features)
    importance = clf.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    
    print('训练损失:',cal_log_loss(train_preds/4, train_Y))
    print('测试损失:',cal_log_loss(cv_preds, train_Y))
    predict_test = np.median(test_preds,axis=1)
    predict_test = predict_test/(predict_test+(1-predict_test)/rate)
    submmit_result(predict_test,'XGB')
    
    return df,clf

if __name__ == '__main__':
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data = load_pickle(path=cache_pkl_path +'test_data')
    
    
    cols = ['user_gender_id','user_age_level','user_occupation_id','user_star_level',\
            'item_brand_id','item_city_id','query_item_second_cate_sim','query_item_second_cate_sim',\
            'user_id_buy_count','item_id_buy_count','item_brand_id_buy_count','shop_id_buy_count',\
            'user_id_cvr_smooth','item_id_cvr_smooth','item_brand_id_cvr_smooth','shop_id_cvr_smooth',\
            'max_cp_cvr','min_cp_cvr','mean_cp_cvr']
    for i in cols:
        train_data[i].replace(to_replace=-1,value=np.nan,inplace=True)
        cv_data[i].replace(to_replace=-1,value=np.nan,inplace=True)
        test_data[i].replace(to_replace=-1,value=np.nan,inplace=True)
        
    drop_cols = ['is_before_dawn','user_id_3day_cvr','user_id_visit#30M','is_afternoon','user_hour_shop_search',\
                 'user_id_2day_cvr','user_id_visit#1H', 'user_id_visit#4H/12H']
    train_data.drop(drop_cols, inplace=True, axis=1)
    cv_data.drop(drop_cols, inplace=True, axis=1)
    test_data.drop(drop_cols, inplace=True, axis=1)
#    train_data.replace(to_replace=-1,value=np.nan,inplace=True)
#    cv_data.replace(to_replace=-1,value=np.nan,inplace=True)
#    test_data.replace(to_replace=-1,value=np.nan,inplace=True)
#    feat_imp, clf = xgb_online(train_data, cv_data, test_data)
    feat_imp, clf = xgb_offline(train_data, cv_data)
    
    t1 = time.time()
    print('训练用时:',t1-t0)
#    plt.figure()
#    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
#    plt.title('XGBoost Feature Importance')  
#    plt.xlabel('relative importance')  
#    plt.show()