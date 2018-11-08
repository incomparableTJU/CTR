#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:39:49 2018

@author: weiqing
"""

from test_1 import load_all_dataset
from autoencoder import autoencoder,autoencoder_add_classify
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

if __name__ == '__main__':
    dataset_dict = load_all_dataset()
    for key in dataset_dict:
        exec('%s = %s["%s"]'%(key,'dataset_dict',key))
    #%%
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
    
    
    #%%
    import pickle
    with open('./new/train_data_cvr_fusion','wb') as f_t:
        pickle.dump(train_data.cvr_fusion,f_t)
        
    with open('./new/test_data_cvr_fusion','wb') as f_t:
        pickle.dump(test_data.cvr_fusion,f_t)
        
    with open('./new/cv_data_cvr_fusion','wb') as f_t:
        pickle.dump(cv_data.cvr_fusion,f_t)
    #%%
    with open('./cache_pkl/train_data','wb') as f_t:
        pickle.dump(train_data,f_t)

    with open('./cache_pkl/test_data','wb') as f_t:
        pickle.dump(test_data,f_t)

    with open('./cache_pkl/cv_data','wb') as f_t:
        pickle.dump(cv_data,f_t)
    #%%
    with open('./cache_pkl/train_data_onehot','wb') as f_t:
        pickle.dump(train_data_onehot,f_t)

    with open('./cache_pkl/test_data_onehot','wb') as f_t:
        pickle.dump(test_data_onehot,f_t)

    with open('./cache_pkl/cv_data_onehot','wb') as f_t:
        pickle.dump(cv_data_onehot,f_t)