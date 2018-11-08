# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:10:36 2018

@author: weiqing
"""

from _1_preprocess import search_category_explore, gen_sorted_search_cate_property
from utils import cache_pkl_path, raw_data_path, feature_data_path, dump_pickle, load_pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from smooth import BayesianSmoothing

# In[]
def get_context_cate_cols(cate_dict, cate_cnt):
    '''
    把根据排序后的类别和属性拼接起来
    '''
    print('generating context cate cols...')
    cols = []
    sorted_cate_items = sorted(cate_dict.items(), key=lambda x: cate_cnt[x[0]], reverse=True)
    for cate in sorted_cate_items:
        cate_cols = list(map(lambda x: str(cate[0])+'_'+str(x), list(cate[1])))
        cols += cate_cols
    return cols
# In[]
def str_to_cate_cols(cate_str):
    '''
    把类别和属性拼接
    '''
    cate_list = cate_str.split(';')
    cate_cols = []
    for cate in cate_list:
        if len(cate.split(':')) < 2:
            continue
        cate_name = cate.split(':')[0]
        cate_value = cate.split(':')[1].split(',')
        cate_cols+=list(map(lambda x: cate_name+'_'+x, cate_value))
    return cate_cols
# In[]
def cal_query_item_second_cate_sim(x):
    '''
    看预测的类别名称是否和商品的类别相匹配
    '''
    second_item_cate = x['item_category_list'].split(';')[1]
    third_item_cate = second_item_cate
    if len(x['item_category_list'].split(';')) > 2:
        third_item_cate = x['item_category_list'].split(';')[2]
    #获取到预测的类别名称
    query_cate = list(map(lambda cate_str: cate_str.split(':')[0], x['predict_category_property'].split(';')))
    find_index = -1
    for idx, cate in enumerate(query_cate):
        if second_item_cate == cate or third_item_cate == cate:
            find_index = idx
            break
    return find_index
# In[]
def cal_query_item_prop_sim(x):
    '''
    预测类别和商品类别不符合时返回-1
    否则返回预测类别属性和商品属性符合的个数
    '''
    if x['query_item_second_cate_sim'] == -1:
        return -1
    second_item_cate = x['item_category_list'].split(';')[1]
    third_item_cate = second_item_cate
    if len(x['item_category_list'].split(';')) > 2:
        third_item_cate = x['item_category_list'].split(';')[2]
    query_cate = list(map(lambda cate_str: cate_str.split(':')[0], x['predict_category_property'].split(';')))    
    query_property = list(map(lambda cate_str: cate_str.split(':')[1], x['predict_category_property'].split(';'))) 
    query_cate_property = dict(zip(query_cate, query_property))
    if second_item_cate in query_cate_property.keys():
        hit_property = query_cate_property[second_item_cate].split(',')
    else:
        hit_property = query_cate_property[third_item_cate].split(',')        
    item_property = x['item_property_list'].split(';')
    return len(set(hit_property).intersection(set(item_property)))

# In[]
def add_query_item_sim(file_name='train'):
    '''
    给原始数据增加2列：
    1、是否预测对了商品类别
    2、预测对类别情况下，属性吻合的个数
    '''
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    
    query_item_info = data[['instance_id', 'item_category_list','item_property_list', 'predict_category_property']]
    data['query_item_second_cate_sim'] = query_item_info.apply(lambda x: cal_query_item_second_cate_sim(x), axis=1)
    data['query_item_prop_sim'] = data.apply(lambda x: cal_query_item_prop_sim(x), axis=1)
    
    data = data[['query_item_second_cate_sim', 'query_item_prop_sim']]
    dump_pickle(data, feature_data_path +file_name + '_query_item_sim')
#    return data
# In[]
def add_context_cate(data):
    
    #得到类别和属性组合后的个数
    context_cate_cols_path = raw_data_path+'context_cate_cols.pkl'
    if os.path.exists(context_cate_cols_path):
        print("found " + context_cate_cols_path)
        cols = load_pickle(context_cate_cols_path)
        cols = list(map(lambda x: x[0], cols))        
    else:
        #cate_dict, cate_cnt, _, _ = search_category_explore(data)
        cols = gen_sorted_search_cate_property(data)
        cols = list(map(lambda x: x[0], cols))
        dump_pickle(cols, context_cate_cols_path)
    
    
    feature_path = feature_data_path + 'context_cate_property_feat.pkl'
    data.cate_cols = data.predict_category_property.apply(lambda x: str_to_cate_cols(x))
    col_index = 0
    #当前商品的类别和属性拼接后是否在前00名
    for col in tqdm(cols[:300]):
        data[col] = data.cate_cols.apply(lambda x: 1 if col in x else 0)
        #if col_index % 200 == 0 and col_index > 100:
        #    dump_pickle(data[['instance_id']+cols[:col_index+1]], feature_path)            
        col_index+=1    
    dump_pickle(data[['instance_id']+cols[:300]], feature_path)
    return data

# In[] 获取类别属性对的转化率
def gen_sorted_cate_property(init_train):
    """
    获取全部的类目-属性对并排序
    """
    cate_col = list(init_train.item_category_list)
    property_col = list(init_train.item_property_list)
    cate_prop_cnt = dict()    
    for cate, properties in zip(cate_col, property_col):
        second_item_cate = cate.split(';')[1]        
        for prop in properties.split(';'):
            cate_prop_col = second_item_cate+'_'+prop
            if cate_prop_col in cate_prop_cnt.keys():
                cate_prop_cnt[cate_prop_col] += 1
            else:
                cate_prop_cnt[cate_prop_col] = 1             
    return sorted(cate_prop_cnt.items(), key=lambda x: x[1], reverse=True)

def gen_cate_property_cvr(test_day, data):
    """
    生成test_day之前全部cate-property对的转化率
    """
    cate_prop_dict_path = cache_pkl_path + 'cate_prop_cvr_day_{0}_dict.pkl'.format(test_day)
    if os.path.exists(cate_prop_dict_path):
        print('found '+cate_prop_dict_path)   
        return load_pickle(cate_prop_dict_path)
    cate_prop_cvr = []
    if test_day != 18:
        real_data = data
        real_data = real_data[real_data['day']<test_day]
        trade_data = real_data[real_data['is_trade']==1]
        all_cate_prop_cnt = gen_sorted_cate_property(real_data)
        trade_cate_prop_cnt = gen_sorted_cate_property(trade_data)
        cate_prop_cvr = trade_cate_prop_cnt
        #平滑滤波
        all_cate_df = pd.DataFrame(all_cate_prop_cnt,columns=['cate_prop','I'])
        trade_cate_df = pd.DataFrame(trade_cate_prop_cnt,columns=['cate_prop','C'])
        all_cate_df = all_cate_df.merge(trade_cate_df, on='cate_prop', how='outer')
        all_cate_df.fillna(0,inplace=True)
        
        hyper = BayesianSmoothing(1, 1)
        hyper.update(all_cate_df['I'].values, all_cate_df['C'].values, 100, 0.00001)
        alpha = hyper.alpha
        beta = hyper.beta
        all_cate_df['cate_prop_cvr_smooth'] = (all_cate_df['C'] + alpha) / (all_cate_df['I'] + alpha + beta)
        
        cate_prop_cvr = all_cate_df[['cate_prop','cate_prop_cvr_smooth']].values
        
#        #不平滑
#        all_cate_prop_cnt = dict(all_cate_prop_cnt)
#        for i, cate_prop in enumerate(cate_prop_cvr):
#            cate_prop_cvr[i] = [cate_prop_cvr[i][0], 1.0*cate_prop[1]/(all_cate_prop_cnt[cate_prop[0]]+1)]
    return cate_prop_cvr

def gen_cate_property_cvr_stats(x, cate_prop_cvr):
    """
    统计cate-property对的转化率，取最大值，最小值，均值
    """
    second_item_cate = x['item_category_list'].split(';')[1]
    properties = x['item_property_list'].split(';')
    cvr_list = []
    for prop in properties:
        cate_prop_col = second_item_cate+'_'+prop
        if cate_prop_col in cate_prop_cvr.keys():
            cvr_list.append(cate_prop_cvr[cate_prop_col])
    if len(cvr_list) == 0:
        return [-1,-1,-1]
    return [max(cvr_list), min(cvr_list), np.mean(cvr_list)]

def add_cate_property_cvr(file_name='train'):
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    cate_prop_cvr_all = None
    if file_name == 'train':
        for day in tqdm((data.day).unique()):
#            cate_prop_dict_path = cache_pkl_path + 'cate_prop_cvr_all_{0}_df.pkl'.format(day)
            cate_prop_cvr = pd.DataFrame()
            cate_prop_cvr_day = gen_cate_property_cvr(day, data)
            cate_prop_cvr_dict = dict(cate_prop_cvr_day)
            cate_prop_feat = data.loc[data.day==day, ['instance_id', 'item_category_list', 'item_property_list']]
            cate_prop_cvr[['max_cp_cvr','min_cp_cvr', 'mean_cp_cvr']] = cate_prop_feat.apply(lambda \
                         x: gen_cate_property_cvr_stats(x, cate_prop_cvr_dict), axis=1)
            cate_prop_cvr_all = pd.concat([cate_prop_cvr_all,cate_prop_cvr], axis=0)
#            dump_pickle(cate_prop_cvr_all, cate_prop_dict_path)
        cate_prop_dict_path = feature_data_path + 'train_cate_prop_cvr'
        dump_pickle(cate_prop_cvr_all, cate_prop_dict_path)
    else:
        train_data = load_pickle(path=raw_data_path + 'train' + '.pkl')
        cate_prop_cvr = pd.DataFrame()
        cate_prop_cvr_day = gen_cate_property_cvr(data.day.min(), train_data)
        cate_prop_cvr_dict = dict(cate_prop_cvr_day)
        cate_prop_feat = data.loc[:, ['instance_id', 'item_category_list', 'item_property_list']]
        cate_prop_cvr[['max_cp_cvr','min_cp_cvr', 'mean_cp_cvr']] = cate_prop_feat.apply(lambda x: gen_cate_property_cvr_stats(x, cate_prop_cvr_dict), axis=1)
        
        cate_prop_dict_path = feature_data_path + 'test_cate_prop_cvr'
        dump_pickle(cate_prop_cvr, cate_prop_dict_path)
    return cate_prop_cvr_all
# In[]
if __name__ == '__main__':
#    train, test = read_init_data()
    
    #train = add_context_cate(train)
#    new_train = add_query_item_sim(train)
    add_query_item_sim('train')
    add_query_item_sim('test')
    
    add_cate_property_cvr('train')
    add_cate_property_cvr('test')