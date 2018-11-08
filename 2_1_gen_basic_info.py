# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:02:51 2018

@author: weiqing
"""

import pandas as pd
import numpy as np
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
from smooth import BayesianSmoothing

'''
生成一些基本特征：
1、用户、商品、店铺的基本信息
2、总值统计：用户、商品、品牌、类目、店铺买入总数
'''
# In[ ]: 获取用户的基本信息，包括：
#id、性别、年龄、职业、星级编号，买入总数，职业是否需要做one-hot?
def gen_user_basic_info(file_name='train',test_day=24):
    data_select = pd.DataFrame()
    
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    data_select['user_id'] = data['user_id']
    data_select['user_gender_id'] = data['user_gender_id']
    data_select['user_age_level'] = data['user_age_level']
    data_select['user_occupation_id'] = data['user_occupation_id']
    data_select['user_star_level'] = data['user_star_level']
    
    #用户搜索时间划分，上午/下午/晚上/凌晨
    data_select['is_morning'] = (data['hour'].values>=8) & (data['hour'].values<=12)
    data_select['is_afternoon'] = (data['hour'].values>12) & (data['hour'].values<=17)
    data_select['is_evening'] = (data['hour'].values>17) & (data['hour'].values<=23)
    data_select['is_before_dawn'] = (data['hour'].values<8)
    
    if file_name == 'train':
        '''
        为了后面的抽样，这里先加上is_trade，训练时记得要删去
        '''
        data_select['is_trade'] = data['is_trade']
    dump_pickle(data_select, feature_data_path +file_name + '_user_basic_info')

# In[]:获取商品的基本特征
def gen_item_basic_info(file_name='train',test_day=24):
    '''
    商品id, 类目、品牌、属性先不考虑、城市、所在展示的页数
    价格等级、销量等级、展示次数、收藏次数
    商品买入总数、品牌买入总数、类目买入总数
    '''
    data_select = pd.DataFrame()
    
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    data_select['item_id'] = data['item_id']
    data_select['second_cate'] = data['second_cate']
    data_select['item_brand_id'] = data['item_brand_id']
    data_select['item_city_id'] = data['item_city_id']
    data_select['item_price_level'] = data['item_price_level']
    data_select['item_sales_level'] = data['item_sales_level']
    data_select['item_pv_level'] = data['item_pv_level']
    data_select['item_collected_level'] = data['item_collected_level']
    data_select['context_page_id'] = data['context_page_id']
    data_select['day'] = data['day']
    
    dump_pickle(data_select, feature_data_path +file_name + '_item_basic_info')
# In[]:生成店铺的基本特征
def gen_shop_basic_features(file_name='train',test_day=24):
    '''
    店铺id、评价数量、好评率、星级、服务态度、物流服务、描述相符等级
    店铺买入总数
    '''
    data_select = pd.DataFrame()
    
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    data_select['shop_id'] = data['shop_id']
    data_select['shop_review_num_level'] = data['shop_review_num_level']
    data_select['shop_review_positive_rate'] = data['shop_review_positive_rate']
    data_select['shop_star_level'] = data['shop_star_level']
    data_select['shop_score_service'] = data['shop_score_service']
    data_select['shop_score_delivery'] = data['shop_score_delivery']
    data_select['shop_score_description'] = data['shop_score_description']
    
    dump_pickle(data_select, feature_data_path +file_name + '_shop_basic_info')
# In[]
def gen_buy_count(file_name='train'):
    
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    cols = ['user_id','item_id','item_brand_id','second_cate','shop_id']
    data_select = pd.DataFrame()
    if file_name == 'train':
        for col in cols:
            feature_str = col + '_buy_count'
            buy_all = None
            for day in data.day.unique():
                buy_filter = data.loc[data.day < day, [col,'is_trade']]
                col_buy_count = buy_filter.groupby([col]).sum().iloc[:,0]
                today_data = data.loc[data.day == day, [col]]
                today_data[feature_str] = today_data.apply(lambda x: \
                  col_buy_count[x[col]] if x[col] in col_buy_count.index else -1, axis=1)
                buy_all = pd.concat([buy_all,today_data], axis=0)
            data_select[feature_str] = buy_all[feature_str]
    else:
        train_data = load_pickle(path=raw_data_path + 'train' + '.pkl')
        for col in cols:
            feature_str = col + '_buy_count'
            buy_filter = train_data.loc[train_data.day <= 24, [col,'is_trade']]
            col_buy_count = buy_filter.groupby([col]).sum().iloc[:,0]
            
            data_select[feature_str] = data.apply(lambda x: \
               col_buy_count[x[col]] if x[col] in col_buy_count.index else -1, axis=1)
    dump_pickle(data_select, feature_data_path +file_name + '_buy_count')
# In[]
if __name__ == '__main__':
    
    gen_user_basic_info('train')
    gen_item_basic_info('train')
    gen_shop_basic_features('train')
    gen_buy_count('train')
    
    gen_user_basic_info('test')
    gen_item_basic_info('test')
    gen_shop_basic_features('test')
    gen_buy_count('test')
    
#    train = load_pickle(path=raw_data_path + 'train' + '.pkl')
#    test = load_pickle(path=raw_data_path + 'test' + '.pkl')
#    
#    x = sorted(train.day.unique())
#    
#    y = ['user_id', 'shop_id', 'item_id']
#    
#    for a in y:
#        print(a)
#        for i in range(1, len(x)):
#            dayj = x[i-1]
#            dayi = x[i]
#            i_user = set(train.loc[train.day==dayi,a].values)
#            j_user = set(train.loc[train.day<=dayj,a].values)
#            
#            uni_user = i_user & j_user
##            print(len(j_user),len(i_user))
#            print('new_'+a+':',len(i_user - uni_user),len(i_user - uni_user)/len(i_user))
#        train_user = set(train.loc[:,a].values)
#        test_user = set(test.loc[:,a].values)
##        print(len(train_user),len(test_user))
#        uni_user = test_user & train_user
#        print('训练和测试集')
#        print('new_'+a+':',len(test_user - uni_user),len(test_user - uni_user)/len(test_user))
        
        