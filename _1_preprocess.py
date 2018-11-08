# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:46:11 2018

@author: weiqing
"""

import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime
import os
from tqdm import tqdm
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
#from utils import read_init_data, addCate

train_file = "round2_train.txt"
test_file = "round2_ijcai_18_test_a_20180425.txt"

# In[ ]:把训练数据和测试数据换成统一的Index

def gen_global_index():
    train = pd.read_table(raw_data_path + train_file,delim_whitespace=True)
    test = pd.read_table(raw_data_path + test_file,delim_whitespace=True)
    all_data = train.append(test)
    all_data['global_index'] = np.arange(0,all_data.shape[0])
    train = all_data.iloc[0:train.shape[0],:]
    test = all_data.iloc[train.shape[0]:,:]
    dump_pickle(train,raw_data_path+'train.pkl')
    dump_pickle(test,raw_data_path+'test.pkl')

# In[ ]: 对时间戳做处理并提取日期和小时

def gen_day_hour(file_name):
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    
    data.context_timestamp = data.context_timestamp.apply(datetime.fromtimestamp)
    data['day'] = data.context_timestamp.apply(lambda x:x.date().day)
    data['hour'] = data.context_timestamp.apply(lambda x:x.time().hour)
    
    dump_pickle(data, path=raw_data_path + file_name + '.pkl')
    
    
# In[]:获得第二个类别
def gen_category(file_name='train'):
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    
    item_cate_col = list(data.item_category_list)
    item_cate = list(map(lambda x: x.split(';'), item_cate_col))
    data['second_cate'] = list(map(lambda x: x[1], item_cate))
    
    dump_pickle(data, path=raw_data_path + file_name + '.pkl')
    
def addCate(file_name='train'):
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    def calSecondCate(x):
        return x['item_category_list'].split(';')[1]
    def calThirdCate(x):
        if len(x['item_category_list'].split(';')) < 3:
            return -1
        return x['item_category_list'].split(';')[2]
    data['second_cate'] = data.apply(lambda x: calSecondCate(x), axis=1)
    data['third_cate'] = data.apply(lambda x: calThirdCate(x), axis=1)
    
    
    dump_pickle(data, path=raw_data_path + file_name + '.pkl')
    
# In[]
def gen_sorted_search_property(init_train):
    '''
    统计所有属性的个数并排序
    '''
    category_property_col = list(init_train.predict_category_property)
    prop_cnt = dict()
    for row in tqdm(category_property_col):
        categories = row.split(';')
        for cate in categories:
            if len(cate.split(':')) < 2:
                continue
            cate_name = int(cate.split(':')[0])
            cate_value = list(map(lambda x: int(x), cate.split(':')[1].split(',')))  
            for prop in cate_value:
                prop_col = str(prop)
                if prop_col in prop_cnt.keys():
                    prop_cnt[prop_col] += 1
                else:
                    prop_cnt[prop_col] = 0
    #return cate_prop_cnt
    return sorted(prop_cnt.items(), key=lambda x: x[1], reverse=True)

# In[]:对类别和属性的统计
def search_category_explore(init_train):
    category_property_col = list(init_train.predict_category_property)
    '''
    1、存放类别下的所有属性
    2、统计所有类别的个数
    3、存放属性对应的所有类别
    4、统计所有属性的个数
    '''
    cate_dict = dict()
    cate_cnt = dict()
    property_dict = dict()
    property_cnt = dict()
    for row in tqdm(category_property_col):
        categories = row.split(';')
        for cate in categories:
            try:
                cate_name = int(cate.split(':')[0])
                cate_value = list(map(lambda x: int(x), cate.split(':')[1].split(',')))
                #print('cate_value', list(cate_value))
                for prop in cate_value:
                    if prop in property_dict.keys():
                        property_dict[prop].add(cate_name)
                        property_cnt[prop] += 1
                    else:
                        property_dict[prop] = set([cate_name])                        
                        property_cnt[prop] = 1
                
                if cate_name in cate_dict.keys():
                    cate_dict[cate_name].update(cate_value)
                    cate_cnt[cate_name] += 1
                else:
                    cate_dict[cate_name] = set(cate_value)
                    cate_cnt[cate_name] = 1
            except:
                print("cate", cate)

    return cate_dict, cate_cnt, property_dict, property_cnt

# In[]:
def gen_sorted_search_cate_property(init_train):
    '''
    统计类别和属性组合后的个数并排序
    '''
    category_property_col = list(init_train.predict_category_property)
    cate_prop_cnt = dict()
    for row in tqdm(category_property_col):
        categories = row.split(';')
        for cate in categories:
            if len(cate.split(':')) < 2:
                continue
            cate_name = int(cate.split(':')[0])
            cate_value = list(map(lambda x: int(x), cate.split(':')[1].split(',')))  
            for prop in cate_value:
                cate_prop_col = str(cate_name)+'_'+str(prop)
                if cate_prop_col in cate_prop_cnt.keys():
                    cate_prop_cnt[cate_prop_col] += 1
                else:
                    cate_prop_cnt[cate_prop_col] = 0
    #return cate_prop_cnt
    return sorted(cate_prop_cnt.items(), key=lambda x: x[1], reverse=True)
# In[]
if __name__ == '__main__':
    
    gen_global_index()
    print("gen_global_index finished")
    gen_day_hour('train')
    print("gen_day_hour finished")
    gen_day_hour('test')
    print("gen_day_hour finished")
    
    addCate('train')
    print("addCate train finished")
    addCate('test')
    print("addCate test finished")
    
    if not os.path.exists(feature_data_path):
        os.mkdir(feature_data_path)
    if not os.path.exists(cache_pkl_path):
        os.mkdir(cache_pkl_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)