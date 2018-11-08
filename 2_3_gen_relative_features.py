# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:36:13 2018

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
生成一些相对特征
'''

# In[]: 获取商品的相对特征
def gen_item_relative_feature(file_name='train'):
    '''
    获取商品相对同类目、品牌平均价格差
    获取商品相对同类目、品牌的平均销量差
    获取商品相对同类目、品牌的平均收藏差
    '''
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    data_select = data.loc[:,['second_cate','item_brand_id','item_price_level','item_sales_level','item_collected_level']]
    item_relative_feature = pd.DataFrame()
    
    #与同类目平均价格的差值
    cate_price_mean= data_select.groupby(['second_cate'])['item_price_level'].mean()
    k = cate_price_mean.loc[data_select.second_cate.values]
    item_relative_feature['cate_relative_price'] = data.item_price_level.values - np.squeeze(k.values)
    
    #与同类目平均销量的差值
    cate_sales_mean= data_select.groupby(['second_cate'])['item_sales_level'].mean()
    k = cate_sales_mean.loc[data_select.second_cate.values]
    item_relative_feature['cate_relative_sales'] = data.item_sales_level.values - np.squeeze(k.values)
    
    #与同类目平均收藏的差值
    cate_collected_mean= data_select.groupby(['second_cate'])['item_collected_level'].mean()
    k = cate_collected_mean.loc[data_select.second_cate.values]
    item_relative_feature['cate_relative_collected'] = data.item_collected_level.values - np.squeeze(k.values)
    
    #与同品牌平均价格的差值
    brand_price_mean= data_select.groupby(['item_brand_id'])['item_price_level'].mean()
    k = brand_price_mean.loc[data_select.item_brand_id.values]
    item_relative_feature['brand_relative_price'] = data.item_price_level.values - np.squeeze(k.values)
    
    #与同品牌平均销量的差值
    brand_sales_mean= data_select.groupby(['item_brand_id'])['item_sales_level'].mean()
    k = brand_sales_mean.loc[data_select.item_brand_id.values]
    item_relative_feature['brand_relative_price'] = data.item_sales_level.values - np.squeeze(k.values)
    
    #与同品牌平均收藏的差值
    brand_collected_mean= data_select.groupby(['item_brand_id'])['item_collected_level'].mean()
    k = brand_collected_mean.loc[data_select.item_brand_id.values]
    item_relative_feature['brand_relative_collected'] = data.item_collected_level.values - np.squeeze(k.values)
    
    dump_pickle(item_relative_feature, feature_data_path +file_name + '_item_relative_feature')
# In[]
if __name__ == '__main__':
    
    gen_item_relative_feature('train')
    gen_item_relative_feature('test')
