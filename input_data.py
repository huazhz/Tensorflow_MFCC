#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:02:35 2017

@author: 390771
"""
import random
import pickle
import numpy as np
def get_data(input_dir):

    with open(input_dir) as f:
        output = pickle.load(f)
    data = output['data']
    target = output['taget']
    NorLabels = np.zeros([len(target),1])
    ErrLabels = np.zeros([len(target),1])
    NorLabels[target==1]=1
    ErrLabels[target==0]=1
    Labels = np.array([NorLabels[:,0],ErrLabels[:,0]]).transpose()
    rng = random.sample(range(len(data)),len(data))
    #0.3 test 0.5 train 0.2 valid
    trainData = data[rng[:623566]]
    testData = data[rng[1247132:]]
    validData = data[rng[623567:1247131]]
    # trainLabels = Labels[rng[:1247131],:]
    # testLabels = Labels[rng[1247132:],:]
    trainLabels = Labels[rng[:623566],:]
    testLabels = Labels[rng[1247132:],:]
    validLabels = Labels[rng[623567:1247131]]
    data_sets = dict()
    data_sets['trainData'] = trainData
    data_sets['testData'] = testData
    data_sets['trainLabels'] = trainLabels
    data_sets['testLabels'] = testLabels

    data_sets['validData'] = validData
    data_sets['validLabels'] = validLabels

    DataSet = {}
    DataSet['train'] = [data_sets['trainData'],data_sets['trainLabels']]
    DataSet['test'] = [data_sets['testData'],data_sets['testLabels']]
    DataSet['valid'] = [data_sets['validData'],data_sets['validLabels']]

    return DataSet

def next_batch(data_sets,batch_size):
    rng = random.sample(range(len(data_sets)),batch_size)
    return data_sets[rng]

if __name__ == '__main__':
    
    input_dir = r'hh.pkl'
    data_sets = get_data(input_dir)
    
           
    