import numpy as np
import os, json, h5py, math, pdb, glob
from PIL import Image
import unicodedata
import cPickle as pkl

options = {}
options['feature_map_len']=[16,8,4,2,1]
options['scale_ratios_anchor1']=[0.25,0.5,0.75,1] #4
options['scale_ratios_anchor2']=[0.25,0.5,0.75,1] #8
options['scale_ratios_anchor3']=[0.25,0.5,0.75,1] #16
options['scale_ratios_anchor4']=[0.25,0.5,0.75,1] #32
options['scale_ratios_anchor5']=[0.25,0.5,0.75,1] #64


SAMPLE_lEN = 64
BATCH_SIZE = 16
max_sen_len = 15

output_path = '../../../data/Charades/h5py/'
train_captions_path = '../../../data/Charades/train.json'
test_captions_path = '../../../data/Charades/test.json'
feature_path = '../../../data/charades_i3d_rgb.hdf5'

train_j = json.load(open(train_captions_path))
test_j = json.load(open(test_captions_path))


def calculate_IOU(groundtruth, predict):
    groundtruth_init = max(0,groundtruth[0])
    groundtruth_end = groundtruth[1]
    predict_init = max(0,predict[0])
    predict_end = predict[1]
    init_min = min(groundtruth_init,predict_init)
    end_max = max(groundtruth_end,predict_end)
    init_max = max(groundtruth_init,predict_init)
    end_min = min(groundtruth_end,predict_end)
    if end_min < init_max:
        return 0
    IOU = ( end_min - init_max ) * 1.0 / ( end_max - init_min)
    return IOU



def generate_anchor(feat_len,feat_ratio,max_len,output_path): # for 64 as an example
    anchor_list = []
    element_span = max_len / feat_len # 1024/64 = 16
    span_list = []
    for kk in feat_ratio:
        span_list.append(kk * element_span)
    for i in range(feat_len): # 64
        inner_list = []
        for span in span_list:
            left =   i*element_span + (element_span * 1 / 2 - span / 2)
            right =  i*element_span + (element_span * 1 / 2 + span / 2) 
            inner_list.append([left,right])
        anchor_list.append(inner_list)
    f = open(output_path,'w')
    f.write(str(anchor_list))
    f.close()
    return anchor_list


def generate_all_anchor():
    all_anchor_list = []
    for i in range(len(options['feature_map_len'])):
        anchor_list = generate_anchor(options['feature_map_len'][i],options['scale_ratios_anchor'+str(i+1)],SAMPLE_lEN,str(i+1)+'.txt')
        all_anchor_list.append(anchor_list)
    return all_anchor_list


def get_anchor_params_unit(anchor,ground_time_step):
    ground_check = ground_time_step[1]-ground_time_step[0]
    if ground_check <= 0:
        return [0.0,0.0,0.0]
    iou = calculate_IOU(ground_time_step,anchor)
    ground_len = ground_time_step[1]-ground_time_step[0]
    ground_center = (ground_time_step[1] - ground_time_step[0]) * 0.5 + ground_time_step[0]
    output_list  = [iou,ground_center,ground_len]
    return output_list


def generate_anchor_params(all_anchor_list,g_position):
    gt_output = np.zeros([len(options['feature_map_len']),max(options['feature_map_len']),len(options['scale_ratios_anchor1'])*3])
    for i in range(len(options['feature_map_len'])): 
        for j in range(options['feature_map_len'][i]): 
            for k in range(len(options['scale_ratios_anchor1'])):
                input_anchor = all_anchor_list[i][j][k]
                output_temp = get_anchor_params_unit(input_anchor,g_position)
                gt_output[i,j,3*k:3*(k+1)]=np.array(output_temp)
    return gt_output


def generate_video_fts_data(video_fts):
    video_fts_shape = np.shape(video_fts)
    video_clip_num = video_fts_shape[0]
    video_fts_dim = video_fts_shape[1]
    output_video_fts = np.zeros([1,SAMPLE_lEN,video_fts_dim])+0.0
    add = 0
    for i in range(video_clip_num):
        if i % 2 == 0 and i+1 <= video_clip_num-1:
            output_video_fts[0,add,:] = np.mean(video_fts[i:i+2,:],0)
            add+=1
        elif i%2 == 0 and i+1 > video_clip_num-1:
            output_video_fts[0,add,:] = video_fts[i,:]
            add+=1
        if add == SAMPLE_lEN:
            return output_video_fts
    print add
    return output_video_fts


def driver(dataset, output_path):
    if dataset == 'train':
        json_data = train_j
    elif dataset == 'test':
        json_data = test_j

    if not os.path.exists(output_path+dataset):
        os.makedirs(output_path+dataset)

    all_video_fts = h5py.File(feature_path,'r')
    all_anchor_list = generate_all_anchor()

    video_source_fts_list = []
    video_names_list = []
    video_duration_list = []
    video_actual_frames_num_list = []
    sentence_list = []
    ground_interval_list = []
    anchor_input_list = []

    cnt = 0
    batch_id = 1
    for video_name in json_data:
        current_data = json_data[video_name]
        video_fts = all_video_fts[video_name]['i3d_rgb_features']
        video_duration = current_data['video_duration']
        for capidx, caption in enumerate(current_data['sentences']):

            anchor_input = generate_anchor_params(all_anchor_list,current_data['timestamps'][capidx])
            video_names_list.append(unicodedata.normalize('NFKD', video_name).encode('ascii','ignore'))
            video_source_fts_list.append(generate_video_fts_data(video_fts))
            video_duration_list.append(video_duration)
            sentence_list.append(unicodedata.normalize('NFKD', caption).encode('ascii','ignore'))
            ground_interval_list.append(current_data['timestamps'][capidx])
            anchor_input_list.append(anchor_input)
            cnt+=1

            if cnt == BATCH_SIZE:
                batch = h5py.File(output_path+'/'+dataset+'/'+dataset+'_'+str(batch_id)+'.h5','w')
                batch['video_source_fts'] = np.array(video_source_fts_list) # batch_size x 512 x 112 x 112 x 3
                batch['video_name'] = np.array(video_names_list) # batch_size
                batch['video_duration'] = np.array(video_duration_list) # batch_size
                batch['sentence'] = np.array(sentence_list) # batch_size
                batch['ground_interval'] = np.array(ground_interval_list) # batch_size x 2
                batch['anchor_input'] = np.array(anchor_input_list)
                print batch_id
                print np.shape(batch['video_source_fts'])
                print np.shape(batch['anchor_input'])
                cnt = 0
                batch_id += 1
                video_source_fts_list = []
                video_names_list = []
                video_duration_list = []
                video_actual_frames_num_list = []
                sentence_list = []
                ground_interval_list = []
                anchor_input_list = []


def getlist(output_path, split):
    List = glob.glob(output_path+'/'+split+'/'+'*.h5')
    f = open(output_path+'/'+split+'/'+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


driver('train', output_path)
getlist(output_path,'train')

driver('test', output_path)
getlist(output_path,'test')

generate_all_anchor()