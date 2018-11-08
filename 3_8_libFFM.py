# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:47:06 2018

@author: weiqing
"""

import pandas as pd
import numpy as np
from utils import load_pickle, raw_data_path, feature_data_path, cache_pkl_path, result_path, model_path,submmit_result
import subprocess
from common import data2libffm
from sklearn.preprocessing import minmax_scale


def binning(series, bin_num):
    bins = np.linspace(series.min(), series.max(), bin_num)
    labels = [i for i in range(bin_num-1)]
    out = pd.cut(series, bins=bins, labels=labels).astype(float)
    return out

def gen_ffm_data_offline(train, cv, test, name):
    columns = ['is_trade'] + fields
    tt = pd.concat([train, cv, test], axis=0)
    for col in continue_field1:
        tt[col] = np.floor(np.log1p(np.abs(tt[col])) ** 2)
    for col in continue_field2:
        tt[col] = binning(tt[col], 5)
    tt = tt[columns]    
    trans_train = tt.iloc[0:len(train), :]
    trans_cv = tt.iloc[len(train):len(train_data)+len(cv_data), :]
    trans_test = tt.iloc[len(train_data)+len(cv_data):, :]
    
    train_path = cache_pkl_path+name+'_train.ffm'
    cv_path = cache_pkl_path+name+'_cv.ffm'
    test_path = cache_pkl_path+name+'_test.ffm'
    
    data2libffm(trans_train, train_path)
    data2libffm(trans_cv, cv_path)
    data2libffm(trans_test, test_path)
    
def gen_ffm_data_online(train, test, name):
    columns = ['is_trade'] + fields
    tt = pd.concat([train, test], axis=0)
    for col in continue_field1:
        tt[col] = np.floor(np.log1p(np.abs(tt[col])) ** 2)
    for col in continue_field2:
        tt[col] = binning(tt[col], 5)
    tt = tt[columns]    
    trans_train = tt.iloc[0:len(train), :]
    trans_test = tt.iloc[len(train_data):, :]
    
    train_path = cache_pkl_path+name+'_train.ffm'
    test_path = cache_pkl_path+name+'_test.ffm'
    
    data2libffm(trans_train, train_path)
    data2libffm(trans_test, test_path)
    
def change_to_result():
    preds = pd.read_csv('../result/ffm_online_result.csv',header=None)
    submmit_result(np.squeeze(preds.values),'FFM')
    
if __name__ == '__main__':
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data = load_pickle(path=cache_pkl_path +'test_data')
    
    cols = ['user_gender_id','user_age_level','user_occupation_id','user_star_level',\
        'item_brand_id','item_city_id','query_item_second_cate_sim','query_item_second_cate_sim',\
        'user_id_buy_count','item_id_buy_count','item_brand_id_buy_count','shop_id_buy_count',\
        'user_id_cvr_smooth','item_id_cvr_smooth','item_brand_id_cvr_smooth','shop_id_cvr_smooth',\
        'max_cp_cvr','min_cp_cvr','mean_cp_cvr']
    for i in cols:
        train_data[i].replace(to_replace=-1,value=0,inplace=True)
        cv_data[i].replace(to_replace=-1,value=0,inplace=True)
        test_data[i].replace(to_replace=-1,value=0,inplace=True)
        
    drop_cols = ['is_before_dawn','user_id_3day_cvr','user_id_visit#30M','is_afternoon','user_hour_shop_search',\
                 'user_id_2day_cvr','user_id_visit#1H', 'user_id_visit#4H/12H']
    train_data.drop(drop_cols, inplace=True, axis=1)
    cv_data.drop(drop_cols, inplace=True, axis=1)
    test_data.drop(drop_cols, inplace=True, axis=1)
    
    categorical_field = ['user_gender_id', 'user_age_level', 'user_occupation_id',
                     'user_star_level', 'is_morning',  'is_evening','item_id',
                     'second_cate', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 
                     'item_pv_level', 'item_collected_level', 'context_page_id', 'day', 'shop_id','shop_review_num_level',
                     'shop_star_level']
    ##连续log平方
    continue_field1 = ['sub_maxtime_item_id', 'sub_mintime_item_id', 'sub_maxtime_shop_id', 
                       'sub_mintime_shop_id','sub_maxtime_item_brand_id', 'sub_mintime_item_brand_id',
                       'sub_maxtime_second_cate', 'sub_mintime_second_cate']
    #连续分箱
#    continue_field2 = ['cate_relative_price', 'cate_relative_sales', 'cate_relative_collected', 'brand_relative_price',\
#                       'brand_relative_collected', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',\
#                       'shop_score_description', 'user_day_cate_search','user_day_item_search','user_day_search',\
#                       'user_day_shop_search','user_hour_cate_search', 'user_hour_item_search','user_hour_search',\
#                       'user_id_buy_count']
    #连续直接当cate
    continue_field2 = list(set(train_data.columns) - set(categorical_field) - set(continue_field1)- set(['is_trade']))
    fields = categorical_field+continue_field1+continue_field2

    gen_ffm_data_offline(train_data, cv_data, test_data, 'offline')
    train_data = pd.concat([train_data, cv_data],axis=0)
    gen_ffm_data_online(train_data, test_data, 'online')