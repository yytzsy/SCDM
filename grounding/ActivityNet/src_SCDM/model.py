"""
Build graph for both training and inference
"""

import tensorflow as tf
from utils import *
slim = tf.contrib.slim
import numpy as np
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
from opt import *


class SSAD_SCDM(object):

    def __init__(self, options, word_emb_init):
        self.options = options
        self.word_emb_init = word_emb_init
        self.anchor_mask_list = self.generate_anchor_mask()
        self.initializer = tf.contrib.layers.xavier_initializer
        self.anchor_attention_0 = {
                "W_v":tf.get_variable("W_v_0",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_0",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_0",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}
        self.anchor_attention_1 = {
                "W_v":tf.get_variable("W_v_1",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_1",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_1",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}
        self.anchor_attention_2 = {
                "W_v":tf.get_variable("W_v_2",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_2",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_2",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}
        self.anchor_attention_3 = {
                "W_v":tf.get_variable("W_v_3",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_3",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_3",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}
        self.anchor_attention_4 = {
                "W_v":tf.get_variable("W_v_4",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_4",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_4",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}
        self.anchor_attention_5 = {
                "W_v":tf.get_variable("W_v_5",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_5",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_5",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}
        self.anchor_attention_6 = {
                "W_v":tf.get_variable("W_v_6",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_6",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_6",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}
        self.anchor_attention_7 = {
                "W_v":tf.get_variable("W_v_7",dtype = tf.float32, shape = (2 * self.options['dim_hidden'], self.options['dim_hidden']), initializer = self.initializer()),
                "W_k":tf.get_variable("W_k_7",dtype = tf.float32, shape = (2 * self.options['dim_hidden'],  self.options['dim_hidden']), initializer = self.initializer()),
                "W_interaction":tf.get_variable("W_interaction_7",dtype = tf.float32, shape = (self.options['dim_hidden']), initializer = self.initializer())}

        self.FULL_ATTENTION_0 = False
        self.FULL_ATTENTION_1 = False

    def generate_anchor(self,feat_len,feat_ratio,max_len):
        anchor_list = []
        element_span = max_len / feat_len 
        span_list = []
        for kk in feat_ratio:
            span_list.append(kk * element_span)
        for i in range(feat_len): 
            inner_list = []
            for span in span_list:
                left =   i*element_span + (element_span * 1 / 2 - span / 2)
                right =  i*element_span + (element_span * 1 / 2 + span / 2) 
                inner_list.append([left,right])
            anchor_list.append(inner_list)
        return anchor_list


    def generate_all_anchor(self):
        all_anchor_list = []
        for i in range(len(self.options['feature_map_len'])):
            anchor_list = self.generate_anchor(self.options['feature_map_len'][i],self.options['scale_ratios_anchor'+str(i+1)],self.options['sample_len'])
            all_anchor_list.append(anchor_list)
        return all_anchor_list


    def generate_anchor_mask(self):
        all_anchor = self.generate_all_anchor()
        length = len(all_anchor)
        output_list = []
        for i in range(length):
            anchor_group_shape = np.shape(all_anchor[i])
            anchor_mask = np.ones([anchor_group_shape[0],anchor_group_shape[1]])+0.0
            for ii in range(anchor_group_shape[0]):
                for jj in range(anchor_group_shape[1]):
                    if all_anchor[i][ii][jj][1] - all_anchor[i][ii][jj][0] <= 1.0:
                        anchor_mask[ii,jj] = 0.0
            anchor_mask = tf.convert_to_tensor(anchor_mask)
            anchor_mask = tf.cast(anchor_mask,tf.float32)
            anchor_mask = tf.expand_dims(tf.expand_dims(anchor_mask,0),1)
            anchor_mask = tf.tile(anchor_mask,[self.options['batch_size'],1,1,1])
            output_list.append(anchor_mask)
        return output_list


    def predict(self,anchor1,is_training,anchor_id):
        predict1_overlap = bn_relu_conv(anchor1,is_training,512,64,1,3,1,name = 'predict%d_1_overlap'%(anchor_id))
        predict1_overlap = bn_relu_conv(predict1_overlap,is_training,64,len(self.options['scale_ratios_anchor%d'%(anchor_id)]),1,3,1,name='predict%d_2_overlap'%(anchor_id))
        predict1_reg = bn_relu_conv(anchor1,is_training,512,64,1,3,1,name = 'predict%d_1_reg'%(anchor_id))
        predict1_reg = bn_relu_conv(predict1_reg,is_training,64,len(self.options['scale_ratios_anchor%d'%(anchor_id)])*self.options['reg_dim'],1,3,1,name='predict%d_2_reg'%(anchor_id))
        predict1_reg = tf.nn.tanh(predict1_reg)
        return predict1_overlap,predict1_reg


    def conv_pool(self, anchor1, sentence_fts, is_training,in_channel,out_channel,h_k,w_k,strides,name_id):
        anchor1 = scdm_relu_conv(anchor1, sentence_fts, is_training, in_channel, out_channel, h_k, w_k, strides, name_id)
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


    def attend_sentence_with_video(self, weights, video_fts, word_squence, word_len, is_training = False, reuse = False, name = 'default_name'):
        with tf.variable_scope('attend_sentence_with_video'+name, reuse = reuse) as scope:
            video_fts_shape = np.shape(video_fts)
            output_attended_sentence_list = []
            for i in range(video_fts_shape[2]):
                if i > 0: tf.get_variable_scope().reuse_variables()
                video_fts_slice = tf.reshape(tf.slice(video_fts,begin=[0,0,i,0],size=[-1,-1,1,-1]),[options['batch_size'],-1])
                inputs = [word_squence, video_fts_slice]
                attention_weights = attention(inputs, self.options['dim_hidden'], weights, scope = "attention_multimodal", memory_len = word_len, reuse = reuse)
                extended_attention_weights = tf.expand_dims(attention_weights, -1)
                attended_sentence_fts = tf.reduce_sum(extended_attention_weights * word_squence, 1)
                output_attended_sentence_list.append(attended_sentence_fts)
            output_sequence = tf.stack(output_attended_sentence_list)
            output_sequence = tf.transpose(output_sequence,[1,0,2])
            return output_sequence


    def fuse_multimodal_feature(self, video_fts, sentence_fts, is_training = False, reuse = False):
        with tf.variable_scope('fuse_multimodal_feature',reuse=reuse) as scope:
            # video_fts: b,1,1024,556
            # sentence_fts: b,512
            sq_video_fts = tf.squeeze(video_fts) # b,1024,556
            video_fts_dim = np.shape(sq_video_fts)
            ep_sentence_fts = tf.tile(tf.expand_dims(sentence_fts,1),[1,video_fts_dim[1],1]) # b,1024,512
            concat_fts = tf.concat([sq_video_fts,ep_sentence_fts],-1) # b,1024,1068
            concat_fts_dim = np.shape(concat_fts)
            concat_fts_reshape = tf.reshape(concat_fts,[-1,concat_fts_dim[2]])
            fused_fts = tf.layers.dense(concat_fts_reshape,self.options['dim_hidden']*2,activation=tf.nn.relu,use_bias=True,reuse=reuse) # b*1024, 512
            fused_fts = tf.reshape(fused_fts,[concat_fts_dim[0],concat_fts_dim[1],-1]) #b, 1024, 512
            fused_fts = tf.expand_dims(fused_fts,1) # b,1,1024,512
            return fused_fts


    def network(self, feature_segment, sentence_fts, word_sequence, word_len, batch_size, is_training=False, reuse=False):
        with tf.variable_scope('base_layer',reuse=reuse) as scope:
            
            multimodal_feature_segment = self.fuse_multimodal_feature(feature_segment,sentence_fts)
            conv1_1 = conv2d(multimodal_feature_segment, self.options['dim_hidden']*2, 512, 1,5, 1, name='conv1')

            if self.FULL_ATTENTION_0:
                # attention mechanism will need large memory
                attention_weights_0 = ([self.anchor_attention_0["W_v"],self.anchor_attention_0["W_k"]],self.anchor_attention_0["W_interaction"])
                sentence_fts_0 = self.attend_sentence_with_video(attention_weights_0, conv1_1, word_sequence, word_len, is_training, reuse, name = 'attend_0')
                conv1 = self.conv_pool(conv1_1,sentence_fts_0,is_training,512,512,1,5,1,0)
                pool1 = tf.nn.max_pool(conv1,ksize=[1,1,2,1], strides=[1,1,2,1],padding='SAME')
            else:
                conv1 = scm_relu_conv(conv1_1,sentence_fts,is_training,512,512,1,5,1,0)
                pool1 = tf.nn.max_pool(conv1,ksize=[1,1,2,1], strides=[1,1,2,1],padding='SAME')


        with tf.variable_scope('anchor_layer',reuse=reuse) as scope:
            attention_weights_1 = ([self.anchor_attention_1["W_v"],self.anchor_attention_1["W_k"]],self.anchor_attention_1["W_interaction"])
            attention_weights_2 = ([self.anchor_attention_2["W_v"],self.anchor_attention_2["W_k"]],self.anchor_attention_2["W_interaction"])
            attention_weights_3 = ([self.anchor_attention_3["W_v"],self.anchor_attention_3["W_k"]],self.anchor_attention_3["W_interaction"])
            attention_weights_4 = ([self.anchor_attention_4["W_v"],self.anchor_attention_4["W_k"]],self.anchor_attention_4["W_interaction"])
            attention_weights_5 = ([self.anchor_attention_5["W_v"],self.anchor_attention_5["W_k"]],self.anchor_attention_5["W_interaction"])
            attention_weights_6 = ([self.anchor_attention_6["W_v"],self.anchor_attention_6["W_k"]],self.anchor_attention_6["W_interaction"])
            attention_weights_7 = ([self.anchor_attention_7["W_v"],self.anchor_attention_7["W_k"]],self.anchor_attention_7["W_interaction"])

            if self.FULL_ATTENTION_1:
                # attention mechanism will need large memory
                sentence_fts_1 = self.attend_sentence_with_video(attention_weights_1, pool1, word_sequence, word_len, is_training, reuse, name = 'attend_1')
                anchor1 = self.conv_pool(pool1,sentence_fts_1,is_training,512,512,1,3,2,1)
            else:
                anchor1 = scm_relu_conv(pool1,sentence_fts,is_training,512,512,1,3,2,1)
            
            sentence_fts_2 = self.attend_sentence_with_video(attention_weights_2, anchor1, word_sequence, word_len, is_training, reuse, name = 'attend_2')
            anchor2 = self.conv_pool(anchor1,sentence_fts_2,is_training,512,512,1,3,2,2)
            
            sentence_fts_3 = self.attend_sentence_with_video(attention_weights_3, anchor2, word_sequence, word_len, is_training, reuse, name = 'attend_3')
            anchor3 = self.conv_pool(anchor2,sentence_fts_3,is_training,512,512,1,3,2,3)
            
            sentence_fts_4 = self.attend_sentence_with_video(attention_weights_4, anchor3, word_sequence, word_len, is_training, reuse, name = 'attend_4')
            anchor4 = self.conv_pool(anchor3,sentence_fts_4,is_training,512,512,1,3,2,4)
            
            sentence_fts_5 = self.attend_sentence_with_video(attention_weights_5, anchor4, word_sequence, word_len, is_training, reuse, name = 'attend_5')
            anchor5 = self.conv_pool(anchor4,sentence_fts_5,is_training,512,512,1,3,2,5)
            
            sentence_fts_6 = self.attend_sentence_with_video(attention_weights_6, anchor5, word_sequence, word_len, is_training, reuse, name = 'attend_6')
            anchor6 = self.conv_pool(anchor5,sentence_fts_6,is_training,512,512,1,3,2,6)

            sentence_fts_7 = self.attend_sentence_with_video(attention_weights_7, anchor6, word_sequence, word_len, is_training, reuse, name = 'attend_7')
            anchor7 = self.conv_pool(anchor6,sentence_fts_7,is_training,512,512,1,3,2,7)

        with tf.variable_scope('prediction_layer',reuse=reuse) as scope:
            predict1_overlap,predict1_reg = self.predict(anchor1,is_training,1) #(b, 1, 256, 4) (b, 1, 256, 8)
            predict2_overlap,predict2_reg = self.predict(anchor2,is_training,2) #(b, 1, 128, 4) (b, 1, 128, 8)
            predict3_overlap,predict3_reg = self.predict(anchor3,is_training,3) #(b, 1, 64, 4) (b, 1, 64, 8)
            predict4_overlap,predict4_reg = self.predict(anchor4,is_training,4) #(b, 1, 32, 4) (b, 1, 32, 8)
            predict5_overlap,predict5_reg = self.predict(anchor5,is_training,5) #(b, 1, 16, 4) (b, 1, 16, 8)
            predict6_overlap,predict6_reg = self.predict(anchor6,is_training,6) #(b, 1, 8, 4) (b, 1, 8, 8)
            predict7_overlap,predict7_reg = self.predict(anchor7,is_training,7) #(b, 1, 4, 4) (b, 1, 4, 8)

        predict_overlap = [predict1_overlap,predict2_overlap,predict3_overlap,predict4_overlap,predict5_overlap,predict6_overlap,predict7_overlap]
        predict_reg = [predict1_reg,predict2_reg,predict3_reg,predict4_reg,predict5_reg,predict6_reg,predict7_reg]
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

        predict_overlap,predict_reg = self.network(feature_segment, sentence_fts, word_sequence, sentence_w_len, batch_size = self.options['batch_size'],is_training=is_training,reuse=reuse)
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

        # (b,7,64,12)
        gt_output = tf.placeholder(tf.float32, [self.options['batch_size'],len(self.options['feature_map_len']),max(self.options['feature_map_len']),4*(1+self.options['reg_dim'])], name='gt_overlap')
        inputs['gt_overlap'] = gt_output

        predict_overlap,predict_reg= self.network(feature_segment, sentence_fts, word_sequence, sentence_w_len, batch_size = self.options['batch_size'],is_training=is_training)

        anchor_predict_loss = []
        positive_loss_list = []
        hard_negative_loss_list = []
        easy_negative_loss_list = []
        smooth_center_loss_list = []
        smooth_width_loss_list = []
        
        for i in range(len(self.options['feature_map_len'])):

            print self.anchor_mask_list[i]

#positive loss
            single_gt_overlap = gt_output[:,i:i+1,:self.options['feature_map_len'][i],:len(self.options['scale_ratios_anchor%d'%(i+1)])*3:3]
            single_gt_overlap_temp = tf.identity(single_gt_overlap)
            ones_now = np.ones([self.options['batch_size'],1,self.options['feature_map_len'][i],len(self.options['scale_ratios_anchor%d'%(i+1)])],np.float32)
            zeros_now=  np.zeros([self.options['batch_size'],1,self.options['feature_map_len'][i],len(self.options['scale_ratios_anchor%d'%(i+1)])],np.float32)
            single_gt_overlap_positive = tf.where(single_gt_overlap_temp>self.options['pos_threshold'],ones_now,zeros_now)
            single_gt_overlap_positive = single_gt_overlap_positive * self.anchor_mask_list[i]
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
            #hard negative is not used which has little improvement.
            single_predict_temp = tf.where(single_predict_temp>self.options['hard_neg_threshold'],ones_now,zeros_now)
            hard_negative_temp = single_predict_temp*single_gt_overlap_negative
            hard_negative_temp = hard_negative_temp * self.anchor_mask_list[i]
            hard_negative_num = tf.reduce_sum(hard_negative_temp)
            hard_negative_loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_overlap[i],labels=single_gt_overlap_temp),hard_negative_temp))
            hard_negative_loss = tf.cond(tf.greater(hard_negative_num,tf.constant(0.0)), lambda: hard_negative_loss/hard_negative_num, lambda: hard_negative_loss)

            easy_negative_temp = single_gt_overlap_negative - hard_negative_temp
            easy_negative_temp = easy_negative_temp * self.anchor_mask_list[i]
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


if __name__ == '__main__':
    options = default_options()
    word_emb_init = np.array(np.load(open(options['word_fts_path'])).tolist(),np.float32)
    model = SSAD_SCDM(options,word_emb_init)
    model.build_train()

    


