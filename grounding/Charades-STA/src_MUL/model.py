"""
Build graph for both training and inference
"""

import tensorflow as tf
from utils import *
slim = tf.contrib.slim
import numpy as np
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
from opt import *


class SSAD_MUL(object):

    def __init__(self, options, word_emb_init):
        self.options = options
        self.initializer = tf.contrib.layers.xavier_initializer
        self.word_emb_init = word_emb_init
        self.fts_attention = {
                "W_v":tf.get_variable("W_v_0",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_0",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_0",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}


    def predict(self,anchor1,is_training,anchor_id):
        predict1_overlap = bn_relu_conv(anchor1,is_training,512,self.options['predict_regression_hidden'],1,3,1,name = 'predict%d_1_overlap'%(anchor_id))
        predict1_overlap = bn_relu_conv(predict1_overlap,is_training,self.options['predict_regression_hidden'],len(self.options['scale_ratios_anchor%d'%(anchor_id)]),1,3,1,name='predict%d_2_overlap'%(anchor_id))
        predict1_reg = bn_relu_conv(anchor1,is_training,512,self.options['predict_regression_hidden'],1,3,1,name = 'predict%d_1_reg'%(anchor_id))
        predict1_reg = bn_relu_conv(predict1_reg,is_training,self.options['predict_regression_hidden'],len(self.options['scale_ratios_anchor%d'%(anchor_id)])*self.options['reg_dim'],1,3,1,name='predict%d_2_reg'%(anchor_id))
        predict1_reg = tf.nn.tanh(predict1_reg)
        return predict1_overlap,predict1_reg


    def conv_pool(self,anchor1,is_training,in_channel,out_channel,h_k,w_k,strides,name_id):
        anchor1 = bn_relu_conv(anchor1, is_training, in_channel, out_channel, h_k,w_k, strides,name='bn_relu_conv'+str(name_id))
        return anchor1


    def encode_sentence(self, sentence_index, sentence_len, is_training = False, reuse = False):
        with tf.variable_scope('sentence_fts',reuse=reuse) as scope:
            with tf.device("/cpu:0"):
                Wemb = tf.Variable(initial_value = self.word_emb_init, name='Wemb')
                sentence_emb = []
                for i in xrange(self.options['max_sen_len']):
                    sentence_emb.append(tf.nn.embedding_lookup(Wemb, sentence_index[:,i]))
                sentence_emb = tf.stack(sentence_emb)
                sentence = tf.transpose(sentence_emb,[1,0,2])
        contextual_word_encoding = bidirectional_GRU(
                                self.options,
                                sentence,
                                sentence_len,
                                units = self.options['dim_hidden'],
                                cell_fn = SRUCell if self.options['SRU'] else GRUCell,
                                layers = self.options['num_layers'],
                                scope = "sentence_encoding",
                                output = 0,
                                is_training = is_training)
        sentence_encoding = avg_sentence_pooling(self.options, contextual_word_encoding, units = self.options['dim_hidden'] , memory_len = sentence_len)
        return contextual_word_encoding, sentence_encoding



    def fuse_multimodal_feature(self, video_fts, sentence_fts, is_training = False, reuse = False):
        with tf.variable_scope('fuse_multimodal_feature',reuse=reuse) as scope:
            # video_fts: b,1,1024,xxxx
            # sentence_fts: b,512
            sq_video_fts = tf.squeeze(video_fts)
            video_fts_dim = np.shape(sq_video_fts)
            ep_sentence_fts = tf.tile(tf.expand_dims(sentence_fts,1),[1,video_fts_dim[1],1])
            concat_fts = tf.concat([sq_video_fts,ep_sentence_fts],-1) 
            concat_fts_dim = np.shape(concat_fts)
            concat_fts_reshape = tf.reshape(concat_fts,[-1,concat_fts_dim[2]])
            fused_fts = tf.layers.dense(concat_fts_reshape,self.options['dim_hidden']*2,activation=tf.nn.relu,use_bias=True,reuse=reuse) # b*1024, 512
            fused_fts = tf.reshape(fused_fts,[concat_fts_dim[0],concat_fts_dim[1],-1]) #b, 1024, 512
            fused_fts = tf.expand_dims(fused_fts,1) # b,1,1024,512
            return fused_fts


    def network(self, feature_segment, sentence_fts, is_training=False, reuse=False):
        with tf.variable_scope('base_layer',reuse=reuse) as scope:
            
            multimodal_feature_segment = self.fuse_multimodal_feature(feature_segment,sentence_fts)

            conv1_1 = conv2d(multimodal_feature_segment, self.options['dim_hidden']*2, 512, 1,5, 1, name='conv1')
            conv1 = bn_relu_conv(conv1_1,is_training,512,512,1,5,1,'bn_relu_conv_conv1')
            sentence_extend = tf.tile(tf.expand_dims(tf.expand_dims(sentence_fts,1),2),[1,1,conv1.shape[2],1])
            conv1 = conv1 * sentence_extend
            pool1 = tf.nn.max_pool(conv1,ksize=[1,1,2,1], strides=[1,1,2,1],padding='SAME')

        with tf.variable_scope('anchor_layer',reuse=reuse) as scope:
            anchor1 = self.conv_pool(pool1,is_training,512,512,1,3,2,1)
            anchor2 = self.conv_pool(anchor1,is_training,512,512,1,3,2,2)
            anchor3 = self.conv_pool(anchor2,is_training,512,512,1,3,2,3)
            anchor4 = self.conv_pool(anchor3,is_training,512,512,1,3,2,4)
            anchor5 = self.conv_pool(anchor4,is_training,512,512,1,3,2,5)

            sentence_extend = tf.tile(tf.expand_dims(tf.expand_dims(sentence_fts,1),2),[1,1,anchor1.shape[2],1])
            anchor1 = anchor1 * sentence_extend

            sentence_extend = tf.tile(tf.expand_dims(tf.expand_dims(sentence_fts,1),2),[1,1,anchor2.shape[2],1])
            anchor2 = anchor2 * sentence_extend  

            sentence_extend = tf.tile(tf.expand_dims(tf.expand_dims(sentence_fts,1),2),[1,1,anchor3.shape[2],1])
            anchor3 = anchor3 * sentence_extend

            sentence_extend = tf.tile(tf.expand_dims(tf.expand_dims(sentence_fts,1),2),[1,1,anchor4.shape[2],1])
            anchor4 = anchor4 * sentence_extend
            
            sentence_extend = tf.tile(tf.expand_dims(tf.expand_dims(sentence_fts,1),2),[1,1,anchor5.shape[2],1])
            anchor5 = anchor5 * sentence_extend

        with tf.variable_scope('prediction_layer',reuse=reuse) as scope:
            predict1_overlap,predict1_reg = self.predict(anchor1,is_training,1)
            predict2_overlap,predict2_reg = self.predict(anchor2,is_training,2)
            predict3_overlap,predict3_reg = self.predict(anchor3,is_training,3)
            predict4_overlap,predict4_reg = self.predict(anchor4,is_training,4)
            predict5_overlap,predict5_reg = self.predict(anchor5,is_training,5)

        predict_overlap = [predict1_overlap,predict2_overlap,predict3_overlap,predict4_overlap,predict5_overlap]
        predict_reg = [predict1_reg,predict2_reg,predict3_reg,predict4_reg,predict5_reg]
        return predict_overlap,predict_reg


    def build_proposal_inference(self,is_training=False,reuse=False):
        inputs={}

        feature_segment = tf.placeholder(tf.float32, [self.options['batch_size'],1,self.options['sample_len'], self.options['video_feat_dim']], name='feature_segment')
        inputs['feature_segment']=feature_segment

        sentence_index_placeholder = tf.placeholder(tf.int32, [self.options['batch_size'],self.options['max_sen_len']])
        sentence_w_len = tf.placeholder(tf.int32, [self.options['batch_size'],])
        word_sequence, sentence_fts = self.encode_sentence(sentence_index_placeholder,sentence_w_len,is_training=is_training)
        inputs['sentence_index_placeholder'] = sentence_index_placeholder
        inputs['sentence_w_len'] = sentence_w_len

        predict_overlap,predict_reg = self.network(feature_segment, sentence_fts, is_training=is_training, reuse=reuse)
        predict_overlap = [tf.nn.sigmoid(i) for i in predict_overlap]
        return inputs,predict_overlap,predict_reg



    def build_train(self,is_training=True):

        inputs={}
        outputs={}

        feature_segment = tf.placeholder(tf.float32, [self.options['batch_size'],1,self.options['sample_len'], self.options['video_feat_dim']], name='feature_segment')
        inputs['feature_segment']=feature_segment

        sentence_index_placeholder = tf.placeholder(tf.int32, [self.options['batch_size'],self.options['max_sen_len']])
        sentence_w_len = tf.placeholder(tf.int32, [self.options['batch_size'],])
        word_sequence, sentence_fts = self.encode_sentence(sentence_index_placeholder,sentence_w_len,is_training=is_training)
        inputs['sentence_index_placeholder'] = sentence_index_placeholder
        inputs['sentence_w_len'] = sentence_w_len

        # (b,len(self.options['feature_map_len']),16,12)
        gt_output = tf.placeholder(tf.float32, [self.options['batch_size'],len(self.options['feature_map_len']),max(self.options['feature_map_len']),4*(1+self.options['reg_dim'])], name='gt_overlap')
        inputs['gt_overlap'] = gt_output

        predict_overlap,predict_reg= self.network(feature_segment, sentence_fts, is_training=is_training)

        anchor_predict_loss = []
        positive_loss_list = []
        hard_negative_loss_list = []
        easy_negative_loss_list = []
        smooth_center_loss_list = []
        smooth_width_loss_list = []
        
        for i in range(len(self.options['feature_map_len'])):
#positive loss
            single_gt_overlap = gt_output[:,i:i+1,:self.options['feature_map_len'][i],:len(self.options['scale_ratios_anchor%d'%(i+1)])*3:3]
            single_gt_overlap_temp = tf.identity(single_gt_overlap)
            ones_now = np.ones([self.options['batch_size'],1,self.options['feature_map_len'][i],len(self.options['scale_ratios_anchor%d'%(i+1)])],np.float32)
            zeros_now=  np.zeros([self.options['batch_size'],1,self.options['feature_map_len'][i],len(self.options['scale_ratios_anchor%d'%(i+1)])],np.float32)
            single_gt_overlap_positive = tf.where(single_gt_overlap_temp>self.options['pos_threshold'],ones_now,zeros_now)
            positive_num = tf.reduce_sum(single_gt_overlap_positive)
            positive_loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_overlap[i],labels=single_gt_overlap_temp),single_gt_overlap_positive))
            positive_loss = tf.cond(tf.greater(positive_num,tf.constant(0.0)), lambda: positive_loss/positive_num, lambda: positive_loss)

#regssion loss
            predict_reg_center = predict_reg[i][:,:,:,::2]
            predict_reg_width = predict_reg[i][:,:,:,1::2]
            set_reg_center = np.zeros([self.options['batch_size'],1,self.options['feature_map_len'][i],len(self.options['scale_ratios_anchor%d'%(i+1)])])
            for j in range(self.options['feature_map_len'][i]):
                set_reg_center[:,:,j,:]=self.options['sample_len']/self.options['feature_map_len'][i]*(j+0.5)
            set_reg_width = np.zeros([self.options['batch_size'],1,self.options['feature_map_len'][i],len(self.options['scale_ratios_anchor%d'%(i+1)])])
            for j in range(len(self.options['scale_ratios_anchor%d'%(i+1)])):
                set_reg_width[:,:,:,j] = self.options['sample_len']*self.options['scale_ratios_anchor%d'%(i+1)][j]/self.options['feature_map_len'][i]
            predict_reg_center = set_reg_center+0.1*set_reg_width*predict_reg_center
            predict_reg_width = set_reg_width*tf.exp(0.1*predict_reg_width)
            gt_center = gt_output[:,i:i+1,:self.options['feature_map_len'][i],1:len(self.options['scale_ratios_anchor%d'%(i+1)])*3:3]
            gt_width = gt_output[:,i:i+1,:self.options['feature_map_len'][i],2:len(self.options['scale_ratios_anchor%d'%(i+1)])*3:3]

            center_min = tf.subtract(predict_reg_center,gt_center)
            center_smooth_sign = tf.cast(tf.less(tf.abs(center_min),1),tf.float32)

            center_smooth_options1 = tf.multiply(center_min,center_min)*0.5
            center_smooth_options2 = tf.subtract(tf.abs(center_min),0.5)
            smooth_center_result = tf.reduce_sum(tf.add(tf.multiply(center_smooth_options1, center_smooth_sign)*single_gt_overlap_positive,
                                  tf.multiply(center_smooth_options2, tf.abs(tf.subtract(center_smooth_sign, 1.0)*single_gt_overlap_positive))))
            smooth_center_result = tf.cond(tf.greater(positive_num,tf.constant(0.0)), lambda: smooth_center_result/positive_num, lambda: smooth_center_result)


            width_min = tf.subtract(predict_reg_width,gt_width)
            width_smooth_sign = tf.cast(tf.less(tf.abs(width_min),1),tf.float32)

            width_smooth_options1 = tf.multiply(width_min,width_min)*0.5
            width_smooth_options2 = tf.subtract(tf.abs(width_min),0.5)
            smooth_width_result = tf.reduce_sum(tf.add(tf.multiply(width_smooth_options1, width_smooth_sign)*single_gt_overlap_positive,
                                  tf.multiply(width_smooth_options2, tf.abs(tf.subtract(width_smooth_sign, 1.0)*single_gt_overlap_positive))))
            smooth_width_result = tf.cond(tf.greater(positive_num,tf.constant(0.0)), lambda: smooth_width_result/positive_num, lambda: smooth_width_result)

#negative loss
            single_gt_overlap_negative = tf.where(single_gt_overlap_temp<self.options['neg_threshold'],ones_now,zeros_now)
            single_predict_temp = tf.identity(predict_overlap[i])
            single_predict_temp = tf.where(single_predict_temp>self.options['hard_neg_threshold'],ones_now,zeros_now)
            hard_negative_temp = single_predict_temp*single_gt_overlap_negative
            hard_negative_num = tf.reduce_sum(hard_negative_temp)
            hard_negative_loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_overlap[i],labels=single_gt_overlap_temp),hard_negative_temp))
            hard_negative_loss = tf.cond(tf.greater(hard_negative_num,tf.constant(0.0)), lambda: hard_negative_loss/hard_negative_num, lambda: hard_negative_loss)

            easy_negative_temp = single_gt_overlap_negative - hard_negative_temp
            easy_negative_num = tf.reduce_sum(easy_negative_temp)
            easy_negative_loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_overlap[i],labels=single_gt_overlap_temp),easy_negative_temp))
            easy_negative_loss = tf.cond(tf.greater(easy_negative_num,tf.constant(0.0)), lambda: easy_negative_loss/easy_negative_num, lambda: easy_negative_loss)

            anchor_predict_loss.append((self.options['posloss_weight']*positive_loss+\
                                        self.options['hardnegloss_weight']*hard_negative_loss+\
                                        self.options['easynegloss_weight']*easy_negative_loss+\
                                        self.options['reg_weight_center']*smooth_center_result+\
                                        self.options['reg_weight_width']*smooth_width_result))
            positive_loss_list.append(self.options['posloss_weight']*positive_loss)
            hard_negative_loss_list.append(self.options['hardnegloss_weight']*hard_negative_loss)
            easy_negative_loss_list.append(self.options['easynegloss_weight']*easy_negative_loss)
            smooth_center_loss_list.append(self.options['reg_weight_center']*smooth_center_result)
            smooth_width_loss_list.append(self.options['reg_weight_width']*smooth_width_result)


        weight_anchor = self.options['weight_anchor']
        for i in range(len(anchor_predict_loss)):
            anchor_predict_loss[i]= anchor_predict_loss[i]*weight_anchor[i]
        loss_ssad = sum(anchor_predict_loss)
        positive_loss_all = sum(positive_loss_list)
        hard_negative_loss_all = sum(hard_negative_loss_list)
        easy_negative_loss_all = sum(easy_negative_loss_list)
        smooth_center_loss_all = sum(smooth_center_loss_list)
        smooth_width_loss_all = sum(smooth_width_loss_list)

        # outputs from proposal module
        outputs['loss_ssad'] = loss_ssad
        outputs['positive_loss_all'] = positive_loss_all
        outputs['hard_negative_loss_all'] = hard_negative_loss_all
        outputs['easy_negative_loss_all'] = easy_negative_loss_all
        outputs['smooth_center_loss_all'] = smooth_center_loss_all
        outputs['smooth_width_loss_all'] = smooth_width_loss_all
        outputs['predict_overlap'] = predict_overlap
        outputs['predict_reg'] = predict_reg

        # L2 regularization
        reg_loss =  tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        outputs['reg_loss'] = self.options['reg'] * reg_loss
        outputs['loss_all'] = outputs['loss_ssad'] + outputs['reg_loss']

        return inputs, outputs


