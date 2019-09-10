"""
Default hyper parameters

Use this setting can obtain similar or even better performance as original SST paper
"""

from collections import OrderedDict
import numpy as np
import sys
import json
import time
import os

def default_options():

    options = OrderedDict()

    options['predict_regression_hidden'] = 64

    #*** MODEL CONFIG ***#
    options['single_feat_dim'] = 1024 
    options['video_feat_dim'] = options['single_feat_dim'] # dim of video feature

    options['feature_map_len']=[16,8,4,2,1]
    options['scale_ratios_anchor1']=[0.25,0.5,0.75,1] #4
    options['scale_ratios_anchor2']=[0.25,0.5,0.75,1] #8
    options['scale_ratios_anchor3']=[0.25,0.5,0.75,1] #16
    options['scale_ratios_anchor4']=[0.25,0.5,0.75,1] #32
    options['scale_ratios_anchor5']=[0.25,0.5,0.75,1] #64
    
    options['reg_dim']=2
    options['weight_anchor'] = [1,1,1,1,1]

    options['batch_size'] = 16      # training batch size
    options['learning_rate'] = 0.0001 # initial learning rate (I fix learning rate to 1e-3 during training phase)
    options['reg'] = 0.001           # regularization strength (control L2 regularization ratio)
    options['max_epochs'] = 200    # maximum training epochs to run
    options['sample_len'] = 64      # the length ratio of the sampled stream compared to the video

    options['posloss_weight'] = 100.0
    options['hardnegloss_weight'] = 50.0
    options['easynegloss_weight'] = 50.0
    options['reg_weight_center'] = 50.0
    options['reg_weight_width'] = 20.0

    options['word_embedding_path'] ='../../../data/glove.840B.300d_dict.npy'
    options['max_sen_len'] = 15
    options['num_layers'] = 1
    options['dim_hidden'] = 256

    options['pos_threshold'] = 0.5
    options['neg_threshold'] = 0.5
    options['hard_neg_threshold'] = 0.1

    options['SRU'] = True
    options['bias'] = True
    options['dropout'] = 0.2
    options['zoneout'] = None

    options['video_data_path_train'] = '../../../data/Charades/h5py/train/train.txt'
    options['video_data_path_test'] = '../../../data/Charades/h5py/test/test.txt'
    options['wordtoix_path'] = '../words/wordtoix.npy'
    options['ixtoword_path'] = '../words/ixtoword.npy'
    options['word_fts_path'] = '../words/word_glove_fts_init.npy'
    options['pre_model_save_dir'] = '../model/'
    options['pre_result_save_dir'] = '../result/'
    options['words_path'] = '../words/'

    options['optimizer'] = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    options['clip'] = True # clip gradient norm
    options['norm'] = 5.0 # global norm
    options['opt_arg'] = {'adam':{'learning_rate':options['learning_rate'], 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8}}
    
    return options

