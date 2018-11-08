# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:48:02 2018

@author: weiqing
"""

import pickle
import pandas as pd
import numpy as np
import scipy.stats as sps
from tqdm import tqdm
import math
import random
import time
import datetime
#file_path
raw_data_path = 'data/'
feature_data_path ='features/'
cache_pkl_path = 'cache_pkl/'
result_path = 'result/'
model_path = 'model/'

def load_pickle(path):
    with open(path,'rb') as f_t:
        return pickle.load(f_t)
def dump_pickle(obj, path, protocol=None,):
    with open(path,'wb') as f_t:
        return pickle.dump(obj,f_t,protocol=protocol)
#    pickle.dump(obj,open(path,'wb'),protocol=protocol)
    
    
def addCate(data):
    def calSecondCate(x):
        return x['item_category_list'].split(';')[1]
    def calThirdCate(x):
        if len(x['item_category_list'].split(';')) < 3:
            return -1
        return x['item_category_list'].split(';')[2]
    data['second_cate'] = data.apply(lambda x: calSecondCate(x), axis=1)
    data['third_cate'] = data.apply(lambda x: calThirdCate(x), axis=1)
    return data

def columns_to_cate(columns):
    cate_index = dict()
    cnt = 0
    for col in columns:
        if not col.rsplit('_', 1)[-1].isalpha():
           if not col.rsplit('_', 1)[0] in cate_index.keys():
               cate_index[col.rsplit('_', 1)[0]] = cnt
               cnt+=1
        else:
            cate_index[col] = cnt
            cnt+=1
    return cate_index
            

def get_hot_key(data, key, threshold):
    key_cnt = data.groupby(key)['instance_id'].count().sort_values(ascending=False).to_frame()
    hot_key = list(key_cnt[key_cnt['instance_id']>threshold].index)
    return hot_key

# def hashstr(input):
#     return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

def gen_hash_row(feats,label,cate_index):
    results = []
    for item in feats:
        col = item.split('#')[0]
        val = item.split('#')[1]
        if val == '0':
            continue
        field_key = col.rsplit('_',1)[0] if not col.rsplit('_',1)[-1].isalpha() else col
        field = cate_index[field_key]
        results.append(gen_hash_item(field, col, val))
    return str(label)+' '+' '.join(results)+'\n'
    
map_col = lambda dat,col: list(map(lambda x: col+'#'+str(x),dat))
gen_hash_item = lambda field, feat, val: '{0}:{1}:{2}'.format(field,hashstr(feat),val)

def data2libffm(merge_dat,output_name):
    merge_dat_val = merge_dat.drop(['is_trade', 'instance_id'],axis=1)
    cols = merge_dat_val.columns
    cate_index = columns_to_cate(cols)
    features = []
    for col in merge_dat_val.columns:
        features.append(map_col(list(merge_dat_val[col]),col))
        #features.appen(merge_dat_val[col].apply())
    #features = np.array(features).T
    #features = list(map(list, zip(*features)))
    label_col = list(merge_dat['is_trade'])
    features_all = []
    with open(output_name,'w') as f_tr:
        for i in tqdm(range(len(features[0]))):
            features_x = [features[x][i] for x in range(len(features))]
            row = gen_hash_row(features_x,label_col[i],cate_index)
            features_all.append(row)
            f_tr.write(row)   
    return features_all

# In[]:采样函数
def split_negative_data(data, n_splits=1):
    negative_data = data[data['is_trade']==0]
    data_length = len(negative_data.index)
    random.seed(0)
    shuffled_index = random.sample(range(data_length), len(list(negative_data.index)))
    splitted_data = None
    batch_length = int(data_length*n_splits)
    
    splitted_data = negative_data.iloc[shuffled_index[0:batch_length],:]
    return splitted_data
        
def build_train_dataset(data, n_splits=1):
    splitted_negative_data = split_negative_data(data, n_splits)
    postive_data = data[data['is_trade']==1]
    train = pd.concat([splitted_negative_data, postive_data],axis=0)
    train = train.sample(frac=1)
#    dump_pickle(train, raw_data_path+'splitted_train.pkl')
    return train
# In[]:损失函数
def cal_log_loss(predict_list, valid_list):
    if len(predict_list) != len(valid_list):
        return -1
    loss = 0
    for predict_label, valid_label in zip(predict_list, valid_list):
        if predict_label <= 0:
            predict_label = 0.00000000001
        if predict_label >= 1:
            predict_label = 0.99999999999
        loss += (valid_label*math.log(predict_label)+(1-valid_label)*math.log(1-predict_label))
    return -loss/(len(predict_list))

# In[]:损失列表
def cal_single_log_loss(predict_list, valid_list):
    if len(predict_list) != len(valid_list):
        return -1
    loss = []
    for predict_label, valid_label in zip(predict_list, valid_list):
        if predict_label <= 0:
            predict_label = 0.00000000001
        if predict_label >= 1:
            predict_label = 0.99999999999
        loss.extend([-1 *(valid_label*math.log(predict_label) + (1-valid_label)*math.log(1-predict_label))])
    return loss

# In[]:
def submmit_result(test_y, name):
    test_file = 'round1_ijcai_18_test_b_20180418.txt'
    test = pd.read_table(raw_data_path + test_file,delim_whitespace=True)
    test_id = test.instance_id
    
    submission = pd.DataFrame({'instance_id':test_id,'predicted_score':test_y})
    submission.to_csv(r'../result/{0}_{1}.txt'.format(name,datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, sep=' ',line_terminator='\r')