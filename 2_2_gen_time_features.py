# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:08:54 2018

@author: weiqing
"""

import pandas as pd
import numpy as np
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
from smooth import BayesianSmoothing
from tqdm import tqdm
'''
生成时间聚合特征和时间差特征
'''

# In[ ]: 用户搜索次数的特征，包括：
#当日搜索特征、目前小时搜索次数
def gen_user_search_count(file_name):
    
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    data = data.loc[:,['user_id', 'item_id', 'shop_id','day', 'hour', 'second_cate']]
    
    data_select = pd.DataFrame()
    
    #聚类一下
    user_day_search = data.groupby(['user_id', 'day']).count().iloc[:,0]
    #获取每个样本的,user_id,day组成的索引，以索引聚类后的数据
    x = data.loc[:, ('user_id', 'day')].values
    k = user_day_search.loc[[tuple(i) for i in x]]
    data_select['user_day_search'] = k.values
    
    user_hour_search = data.groupby(['user_id', 'day', 'hour']).count().iloc[:,0]
    x = data.loc[:, ('user_id', 'day', 'hour')].values
    k = user_hour_search.loc[[tuple(i) for i in x]]
    data_select['user_hour_search'] = k.values
    
    user_day_item_search = data.groupby(['user_id', 'day', 'item_id']).count().iloc[:,0]
    x = data.loc[:, ('user_id', 'day', 'item_id')].values
    k = user_day_item_search.loc[[tuple(i) for i in x]]
    data_select['user_day_item_search'] = k.values
    
    user_hour_item_search = data.groupby(['user_id', 'day', 'hour', 'item_id']).count().iloc[:,0]
    x = data.loc[:, ('user_id', 'day', 'hour', 'item_id')].values
    k = user_hour_item_search.loc[[tuple(i) for i in x]]
    data_select['user_hour_item_search'] = k.values
    
    user_day_shop_search = data.groupby(['user_id', 'day', 'shop_id']).count().iloc[:,0]
    x = data.loc[:, ('user_id', 'day', 'shop_id')].values
    k = user_day_shop_search.loc[[tuple(i) for i in x]]
    data_select['user_day_shop_search'] = k.values
    
    user_hour_shop_search = data.groupby(['user_id', 'day', 'hour', 'shop_id']).count().iloc[:,0]
    x = data.loc[:, ('user_id', 'day', 'hour', 'shop_id')].values
    k = user_hour_shop_search.loc[[tuple(i) for i in x]]
    data_select['user_hour_shop_search'] = k.values
    
    
    user_day_catesearch = data.groupby(['user_id', 'day', 'second_cate']).count().iloc[:,0]
    x = data.loc[:, ('user_id', 'day', 'second_cate')].values
    k = user_day_catesearch.loc[[tuple(i) for i in x]]
    data_select['user_day_cate_search'] = k.values
    
    user_hour_cate_search = data.groupby(['user_id', 'day', 'hour', 'second_cate']).count().iloc[:,0]
    x = data.loc[:, ('user_id', 'day', 'hour', 'second_cate')].values
    k = user_hour_cate_search.loc[[tuple(i) for i in x]]
    data_select['user_hour_cate_search'] = k.values
    
    dump_pickle(data_select, feature_data_path +file_name + '_user_search_count')
    
    
# In[]:生成用户的时间差特征：
def gen_user_search_time(file_name):
    '''
    #用当次搜索距离当天第一次搜索该商品时间差
    #用当次搜索距离当天第最后一次搜索该商品时间差
    #用当次搜索距离当天第一次搜索该店铺时间差
    #用当次搜索距离当天第最后一次搜索该店铺时间差
    #用当次搜索距离当天第一次搜索该品牌时间差
    #用当次搜索距离当天第最后一次搜索该品牌时间差
    #用当次搜索距离当天第一次搜索该类目时间差
    #用当次搜索距离当天第最后一次搜索该类目时间差
    '''
    data_select = pd.DataFrame()
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    
    cols = ['item_id','shop_id', 'item_brand_id','second_cate']
    for col in cols:
        data_filter = data[['user_id', col,'day','context_timestamp']].groupby(['user_id', col,'day'])
        max_time = data_filter.agg(max)
        min_time = data_filter.agg(min)
        x = data.loc[:, ('user_id', col, 'day')].values
        m = max_time.loc[[tuple(i) for i in x]]
        n = min_time.loc[[tuple(i) for i in x]]
        data_select['sub_maxtime_'+col] = data['context_timestamp'].values - np.squeeze(m.values)
        data_select['sub_mintime_'+col] = data['context_timestamp'].values - np.squeeze(n.values)
        
        data_select['sub_maxtime_'+col] = data_select['sub_maxtime_'+col].apply(lambda x: x.total_seconds())
        data_select['sub_mintime_'+col] = data_select['sub_mintime_'+col].apply(lambda x: x.total_seconds())
    dump_pickle(data_select, feature_data_path +file_name + '_user_search_time')
    

# In[]
if __name__ == '__main__':
    
    gen_user_search_count('train')
    gen_user_search_time('train')
    
    gen_user_search_count('test')
    gen_user_search_time('test')