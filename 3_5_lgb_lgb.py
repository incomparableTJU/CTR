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

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle,build_train_dataset,cal_log_loss,cal_single_log_loss
from smooth import BayesianSmoothing
import gen_smooth_features as smooth_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
import lightgbm as lgb
import operator 
import matplotlib.pyplot as plt

params = {
    'max_depth': 4,                 #3
#    'min_data_in_leaf': 2,
    'feature_fraction': 1,       #0.45
    'learning_rate': 0.04,          #0.16
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'verbose': -1,
    'metric': 'binary_logloss',
}

if __name__ == '__main__':
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data_onehot')
    train_Y = load_pickle(path=cache_pkl_path +'train_Y')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data_onehot')
    cv_Y = load_pickle(path=cache_pkl_path +'cv_Y')
    
    test_data = load_pickle(path=cache_pkl_path +'test_data_onehot')
    test_file = 'round1_ijcai_18_test_a_20180301.txt'
    test = pd.read_table(raw_data_path + test_file,delim_whitespace=True)
    test_id = test.instance_id
    
    train_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    cv_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    test_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    
    
    print('train shap:',train_data.shape)
    print('cv shape', cv_data.shape)
    print('test shape', test_data.shape)
    
    lgb_train = lgb.Dataset(train_data.values, train_Y)
    lgb_cv = lgb.Dataset(cv_data.values, cv_Y)
    gbm = lgb.train(params=params,            #参数
                    train_set=lgb_train,      #要训练的数据
                    num_boost_round=6000,     #迭代次数
                    valid_sets=lgb_cv,        #训练时需要评估的列表
                    verbose_eval=False,       #
                    early_stopping_rounds=500)
    
    predict_train_fir = gbm.predict(train_data.values)
    predict_cv_fir = gbm.predict(cv_data.values)
    predict_test_fir = gbm.predict(test_data.values)
    
    print('训练损失:',cal_log_loss(predict_train_fir, train_Y))
    print('测试损失:',cal_log_loss(predict_cv_fir, cv_Y))
    
    #把全量数据拿过来训练
    train_data_all = pd.concat([train_data, cv_data],axis=0)
    train_Y_all = np.append(train_Y, cv_Y)
    
    lgb_train = lgb.Dataset(train_data_all.values, train_Y_all)
    gbm = lgb.train(params=params,            #参数
                    train_set=lgb_train,      #要训练的数据
                    num_boost_round=500,     #迭代次数
                    verbose_eval=False)
    print('训练损失:',cal_log_loss(gbm.predict(train_data_all.values), train_Y_all))
    predict_test_fir = gbm.predict(test_data.values)
    
    submission = pd.DataFrame({'instance_id':test_id,'predicted_score':predict_test_fir})
    print('预测正样本比例:',len(submission.loc[submission.predicted_score>=0.5])/len(submission))
    submission.to_csv(r'../result/LGB_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, sep=' ',line_terminator='\r')
    #%% 
    '''
    粗筛后删除分数较低的那些样本
    '''
    #按一定比例抽取概率较低的数据，作为LR的结果
    precent_2 = 0.95
    LR_cv_len = int(len(cv_data)*precent_2)
    LR_test_len = int(len(test_data)*precent_2)
    
    #LR的较低的分数直接拼接到最后的结果中
    predict_cv_df = pd.DataFrame({'col_index':cv_data.index,'score':predict_cv_fir})
    predict_cv_df.sort_values(by='score',inplace=True)
    predict_test_df = pd.DataFrame({'col_index':test_data.index,'score':predict_test_fir})
    predict_test_df.sort_values(by='score',inplace=True)
    
    LR_cv_score = predict_cv_df.iloc[0:LR_cv_len]
    LR_test_score = predict_test_df.iloc[0:LR_test_len]
    
    LGB_cv_index = predict_cv_df.iloc[LR_cv_len:]['col_index'].values
    LGB_test_index = predict_test_df.iloc[LR_test_len:]['col_index'].values
    
    #较难训练的样本的index 采用LGBoost来训练
    precent_1 = 0.65
    drop_len = int(len(train_data)*precent_1)
    
    predict_train_df = pd.DataFrame({'col_index':train_data.index,'score':predict_train_fir})
    predict_train_df.sort_values(by='score',inplace=True)
    
    #概率排名较低的前drop_len个样本中，如果label是0，则删除
    drop_index = predict_train_df.col_index[0:drop_len].apply(lambda x:(train_Y[x]==0))
    drop_index = drop_index[drop_index==True].index
    
    LR_train_score = predict_train_df.loc[drop_index]
    #得到保留下来的，较难训练的样本
    predict_train_df.drop(drop_index, inplace=True)
    reserved_index  = predict_train_df.col_index.values
    
    
    #读取LGBoost特征，训练数据
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data = load_pickle(path=cache_pkl_path +'test_data')
    
    train_data_sec = train_data.loc[reserved_index,:]
    train_Y_sec = train_Y[reserved_index]
    
    
    lgb_train = lgb.Dataset(train_data_sec.values, train_Y_sec)
    lgb_cv = lgb.Dataset(cv_data.values, cv_Y)
    gbm = lgb.train(params=params,            #参数
                    train_set=lgb_train,      #要训练的数据
                    num_boost_round=6000,     #迭代次数
                    valid_sets=lgb_cv,        #训练时需要评估的列表
                    verbose_eval=False,       #
                    
                    early_stopping_rounds=500)
    
    predict_train_sec = gbm.predict(train_data_sec.values)
    predict_cv_sec = gbm.predict(cv_data.values)
    predict_test_sec = gbm.predict(test_data.values)
    
    LGB_train_score = pd.DataFrame({'col_index':train_data_sec.index,'score':predict_train_sec})
    LGB_cv_score = pd.DataFrame({'col_index':cv_data.index,'score':predict_cv_sec})
    LGB_cv_score = LGB_cv_score.loc[LGB_cv_index]
    LGB_test_score = pd.DataFrame({'col_index':test_data.index,'score':predict_test_sec})
    LGB_test_score = LGB_test_score.loc[LGB_test_index]
    
    #将LR和LGBoost的结果混合
    train_all = pd.concat([LR_train_score, LGB_train_score],axis=0)
    train_all.sort_values(by='col_index',inplace=True)
    cv_all = pd.concat([LR_cv_score, LGB_cv_score],axis=0)
    cv_all.sort_values(by='col_index',inplace=True)
    test_all = pd.concat([LR_test_score, LGB_test_score],axis=0)
    test_all.sort_values(by='col_index',inplace=True)
    
    print('训练损失:',cal_log_loss(train_all.score.values, train_Y))
    print('测试损失:',cal_log_loss(cv_all.score.values, cv_Y))
    
    t1 = time.time()
    print('训练用时：',t1-t0)
    
    
    submission = pd.DataFrame({'instance_id':test_id,'predicted_score':test_all.score.values})
    print('预测正样本比例:',len(submission.loc[submission.predicted_score>=0.5])/len(submission))
    submission.to_csv(r'../result/LGB_LGB_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, sep=' ',line_terminator='\r')
    
    