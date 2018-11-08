# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:46:04 2018

@author: weiqing
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
import random
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle,build_train_dataset,cal_log_loss
from smooth import BayesianSmoothing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
from autoencoder import autoencoder,autoencoder_add_classify

 # In[]:读取特征并整合成训练数据和测试数据
def gen_train_data(file_name='train', test_day=24):
    
    data = pd.DataFrame()
    
    #读取做好的特征文件
    user_basic_info = load_pickle(path=feature_data_path + file_name + '_user_basic_info')
    user_search_count = load_pickle(path=feature_data_path + file_name + '_user_search_count')
    user_search_time = load_pickle(path=feature_data_path + file_name + '_user_search_time')
    
    item_basic_info = load_pickle(path=feature_data_path + file_name + '_item_basic_info')
    item_relative_info = load_pickle(path=feature_data_path + file_name + '_item_relative_feature')
    query_item_sim = load_pickle(path=feature_data_path + file_name + '_query_item_sim')
    
    shop_basic_info = load_pickle(path=feature_data_path + file_name + '_shop_basic_info')
    
    buy_count = load_pickle(path=feature_data_path + file_name + '_buy_count')
    cvr_smooth = load_pickle(path=feature_data_path + file_name + '_cvr_smooth')
    cate_prop_cvr = load_pickle(path=feature_data_path + file_name + '_cate_prop_cvr')
    
    cols = ['user_id','item_id','shop_id','second_cate',]
    for col in  cols:
        cvr_a = load_pickle(path=feature_data_path + file_name + col + 'cvr_day')
        data = pd.concat([data, cvr_a],axis=1)
    
    data = pd.concat([data, user_basic_info,user_search_count,user_search_time,\
                      item_basic_info,item_relative_info,query_item_sim,shop_basic_info,\
                      buy_count,cvr_smooth,cate_prop_cvr],axis=1)
    
    #对以下列做onehot
#    cols = ['user_gender_id','user_age_level','user_occupation_id','user_star_level']
#    for col in cols:
#        temp_col = pd.get_dummies(data[col],prefix=col)
#        data.drop(col, axis=1, inplace=True)
#        data = pd.concat([data, temp_col],axis=1)
    #对以下列做填充
    cols = ['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']
    for col in cols:
        data[col] = data[col].replace(to_replace=-1,value=data[col].median())
    #7号都是缺失值，可以考虑删除
#    data.drop(data.index[data.day==17],inplace=True, axis=0)
    #把销量、价格、收藏次数以下特征取对数
    data['item_sales_level'].replace(to_replace=-1,value=0,inplace=True)
    cols = ['item_sales_level','item_collected_level','item_pv_level']
    for col in cols:
        data[col] = np.log1p(data[col])
        
    if file_name == 'train':
        data = data.drop_duplicates()
        view_data = load_pickle(path=feature_data_path + file_name + '_visit')
        data = pd.concat([data, view_data],axis=1)
        #划分训练数据和测试数据
#        data_length = len(data.index)
#        shuffled_index = random.sample(range(data_length), data_length)
#        cv_length = int(data_length*0.2)
#        
#        cv_data = data.iloc[shuffled_index[0:cv_length],:]
#        train_data = data.iloc[shuffled_index[cv_length:],:]
        train_data = data.loc[data.day<test_day,:].copy()
        cv_data = data.loc[data.day>=test_day,:].copy()
        
        #对训练数据的负样本进行1/7的采样
#        train_data = build_train_dataset(train_data)
        train_Y = train_data.is_trade.values
#        train_data.drop(['is_trade'],axis=1,inplace=True)
        
        test_Y = cv_data.is_trade.values
#        cv_data.drop(['is_trade'],axis=1,inplace=True)
        
        cv_data.reset_index(inplace=True,drop=True)
        train_data.reset_index(inplace=True,drop=True)
        #保存文件
        dump_pickle(train_data, cache_pkl_path +'train_data')
        dump_pickle(train_Y, cache_pkl_path +'train_Y')
        dump_pickle(cv_data, cache_pkl_path +'cv_data')
        dump_pickle(test_Y, cache_pkl_path +'cv_Y')
    else:
        view_data = load_pickle(path=feature_data_path + file_name + '_visit')
        data = pd.concat([data, view_data],axis=1)
        data['is_trade'] = 0
        data.reset_index(inplace=True,drop=True)
        dump_pickle(data, cache_pkl_path +'test_data')
    
def gen_one_hot_data():
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data= load_pickle(path=cache_pkl_path +'test_data')
    
    cv_data.index += len(train_data)
    test_data.index += len(train_data) + len(cv_data)
    
    data = pd.concat([train_data,cv_data, test_data],axis=0)
    
    #对非线性的cvr进行分段处理
#    cols_divide = {'user_id_cvr_smooth':0.05,'item_id_cvr_smooth':0.075,\
#                   'item_brand_id_cvr_smooth':0.075,'second_cate_cvr_smooth':0.045,\
#                   'shop_id_cvr_smooth':0.075,'max_cp_cvr':0.1,'min_cp_cvr':0.04,'mean_cp_cvr':0.05}
#    
#    for key, value in cols_divide.items():
#        str_col = key+'_-1'
#        data[str_col] = data[key] == -1
#        str_col = key+'_sma'+str(value)
#        data[str_col] = (data[key] != -1) & (data[key]<=value)
#        str_col = key+'_gra'+str(value)
#        data[str_col] = (data[key] != -1) & (data[key]>value)
#        data.drop(key, axis=1, inplace=True)
    
    cols = ['user_gender_id','user_age_level','user_occupation_id','user_star_level',\
        'item_brand_id','item_city_id','query_item_second_cate_sim','query_item_second_cate_sim',\
        'user_id_buy_count','item_id_buy_count','item_brand_id_buy_count','shop_id_buy_count',\
        'user_id_cvr_smooth','item_id_cvr_smooth','item_brand_id_cvr_smooth','shop_id_cvr_smooth',\
        'max_cp_cvr','min_cp_cvr','mean_cp_cvr']
    for i in cols:
        data[i].replace(to_replace=-1,value=0,inplace=True)
    cols = ['second_cate','item_price_level','item_city_id'
            ,'context_page_id','shop_review_num_level']
    
    for col in cols:
        col_feature = pd.get_dummies(data[col], prefix=col)
        data.drop([col],axis=1,inplace=True)
        data = pd.concat([data,col_feature], axis=1)
    
    X = minmax_scale(data.values)
    data = pd.DataFrame(data=X, columns=data.columns)
    
    train_data = data.loc[train_data.index]
    cv_data = data.loc[cv_data.index]
    test_data = data.loc[test_data.index]
    
    
    train_data.reset_index(inplace=True,drop=True)
    cv_data.reset_index(inplace=True,drop=True)
    test_data.reset_index(inplace=True,drop=True)
    
    dump_pickle(train_data, cache_pkl_path +'train_data_onehot')
    dump_pickle(cv_data, cache_pkl_path +'cv_data_onehot')
    dump_pickle(test_data, cache_pkl_path +'test_data_onehot')
    
# In[]:把多个cvr_smooth拟合成一个
def gen_cvr_fusion():
    train_data_onehot = load_pickle(path=cache_pkl_path +'train_data_onehot')
    cv_data_onehot = load_pickle(path=cache_pkl_path +'cv_data_onehot')
    test_data_onehot = load_pickle(path=cache_pkl_path +'test_data_onehot')
    
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data = load_pickle(path=cache_pkl_path +'test_data')
    
    all_data = pd.concat([train_data,cv_data,test_data])
    cvr_cols = ['user_id_cvr_smooth',
                 'item_id_cvr_smooth',
                 'item_brand_id_cvr_smooth',
                 'second_cate_cvr_smooth',
                 'shop_id_cvr_smooth',]
    cvr_data = all_data[cvr_cols]
    cvr_data_min_max = (cvr_data-cvr_data.min()) /(cvr_data.max()-cvr_data.min())
    
    #%% 
    a = autoencoder(input_size = len(cvr_cols),hidden_layers=[16,8,4,1])
    #a.load_model()
    a.train(cvr_data_min_max,5000)
    new_cvr = a.get_feature(cvr_data_min_max)
    new_cvr = new_cvr.reshape(-1)
    train_data['cvr_fusion'] = new_cvr[:len(train_data)]
    cv_data['cvr_fusion'] = new_cvr[len(train_data):len(train_data)+len(cv_data)]
    test_data['cvr_fusion'] = new_cvr[len(train_data)+len(cv_data):len(train_data)+len(cv_data)+len(test_data)]
    train_data_onehot['cvr_fusion'] = train_data.cvr_fusion
    test_data_onehot['cvr_fusion'] = test_data.cvr_fusion
    cv_data_onehot['cvr_fusion'] = cv_data.cvr_fusion
    
    dump_pickle(train_data, cache_pkl_path +'train_data')
    dump_pickle(cv_data, cache_pkl_path +'cv_data')
    dump_pickle(test_data, cache_pkl_path +'test_data')
    
    dump_pickle(train_data_onehot, cache_pkl_path +'train_data_onehot')
    dump_pickle(cv_data_onehot, cache_pkl_path +'cv_data_onehot')
    dump_pickle(test_data_onehot, cache_pkl_path +'test_data_onehot')
if __name__ == '__main__':
    
    gen_train_data('train')
    gen_train_data('test')
    gen_one_hot_data()
    gen_cvr_fusion()