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

    options['single_feat_dim'] = 4096
    options['video_feat_dim'] = options['single_feat_dim'] # dim of video feature

    options['feature_map_len']=[256,128,64,32,16,8,4]
    options['scale_ratios_anchor1']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor2']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor3']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor4']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor5']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor6']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor7']=[0.25,0.5,0.75,1]
    
    options['reg_dim']=2
    options['weight_anchor'] = [1,1,1,1,1,1,1]

    options['use_weight'] = True     # whether use pre-calculated weights for positive/negative samples (deal with imbalance class problem)
    options['batch_size'] = 16      # training batch size
    options['learning_rate'] = 0.0001 # initial learning rate (I fix learning rate to 1e-3 during training phase)
    options['reg'] = 0.0001           # regularization strength (control L2 regularization ratio)
    options['init_scale'] = 0.08     # the init scale for uniform distribution
    options['max_epochs'] = 250    # maximum training epochs to run
    options['init_epoch'] = 0        # initial epoch (useful when you needs to continue from some checkpoints)
    options['n_eval_per_epoch'] = 0.5 # number of evaluations per epoch
    options['eval_init'] = False     # whether to evaluate the initialized model
    options['shuffle'] = True        # whether do data shuffling for training set
    options['log_input_min']  = 1e-20          # minimum input to the log() function
    options['sample_len'] = 1024       # the length ratio of the sampled stream compared to the video
    options['proposal_tiou_threshold'] = 0.5   # tiou threshold to generate positive samples, when changed, re-calculate class weights for positive/negative class
    options['n_iters_display'] = 1             # display frequency

    options['posloss_weight'] = 100.0
    options['hardnegloss_weight'] = 50.0
    options['easynegloss_weight'] = 50.0
    options['reg_weight_center'] = 1.0
    options['reg_weight_width'] = 1.0

    options['pos_threshold'] = 0.5
    options['neg_threshold'] = 0.3
    options['hard_neg_threshold'] = 2.0

    options['word_embedding_path'] ='../../../data/glove.840B.300d_dict.npy'
    options['max_sen_len'] = 35
    options['num_layers'] = 1
    options['dim_hidden'] = 256

    options['SRU'] = True
    options['bias'] = True
    options['dropout'] = 0.2
    options['zoneout'] = None

    options['video_fts_path'] = '../../../data/activitynet_c3d_fc6_stride_1s.hdf5'
    options['video_data_path_train'] = '../../../data/ActivityNet/h5py/train/train.txt'
    options['video_data_path_val'] = '../../../data/ActivityNet/h5py/val/val.txt'
    options['wordtoix_path'] = '../words/wordtoix.npy'
    options['ixtoword_path'] = '../words/ixtoword.npy'
    options['word_fts_path'] = '../words/word_glove_fts_init.npy'
    options['model_save_dir'] = '../model/'
    options['result_save_dir'] = '../result/'
    options['words_path'] = '../words/'
    options['video_info_path'] = '../../../data/ActivityNet/data_info/video_info.pkl'
 
    options['optimizer'] = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    options['clip'] = True # clip gradient norm
    options['norm'] = 5.0 # global norm
    options['opt_arg'] = {'adam':{'learning_rate':options['learning_rate'], 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8}}
    
    return options

