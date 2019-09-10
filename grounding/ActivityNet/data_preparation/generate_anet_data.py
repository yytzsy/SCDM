import numpy as np
import os, json, h5py, math, pdb, glob
from PIL import Image
import unicodedata
import cPickle as pkl

options = {}
options['feature_map_len']=[256,128,64,32,16,8,4]
options['scale_ratios_anchor1']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor2']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor3']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor4']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor5']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor6']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor7']=[0.25,0.5,0.75,1]



SAMPLE_lEN = 1024
BATCH_SIZE = 16



output_path = '../../../data/ActivityNet/h5py/'

splitdataset_path = '../../../data/ActivityNet/data_info/ActivityNet_dataset_split_full.npz'
train_captions_path = '../../../data/ActivityNet/data_info/train.json'
val_captions_path = '../../../data/ActivityNet/data_info/val_merge.json'
video_info_path = '../../../data/ActivityNet/data_info/video_info.pkl'


video_info = pkl.load(open(video_info_path))
train_j = json.load(open(train_captions_path))
val_j = json.load(open(val_captions_path))


def get_video_info(video_name):
    if video_name not in video_info:
        return -1, -1
    else:
        content = video_info[video_name]
        fps = content[0]
        frame_num = content[1]
        return fps, frame_num


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
    gt_output = np.zeros([len(options['feature_map_len']),max(options['feature_map_len']),len(options['scale_ratios_anchor1'])*3]) #[7,64,4*(1+2)]
    for i in range(len(options['feature_map_len'])):
        for j in range(options['feature_map_len'][i]): 
            for k in range(len(options['scale_ratios_anchor1'])):
                input_anchor = all_anchor_list[i][j][k]
                output_temp = get_anchor_params_unit(input_anchor,g_position)
                gt_output[i,j,3*k:3*(k+1)]=np.array(output_temp)
    return gt_output



def driver(dataset, output_path):
    if dataset == 'train':
        info = train_j
    elif dataset == 'val':
        info = val_j

    if not os.path.exists(output_path+dataset):
        os.makedirs(output_path+dataset)

    all_anchor_list = generate_all_anchor()

    video_names_list = []
    video_duration_list = []
    video_actual_frames_num_list = []
    sentence_list = []
    ground_interval_list = []
    anchor_input_list = []

    cnt = 0
    batch_id = 1
    List = np.load(splitdataset_path)[dataset] 
    for iii in range(len(List)):
        video_fps, video_frames_num = get_video_info(List[iii])
        if video_fps == -1 or video_frames_num == -1:
            continue
        for capidx, caption in enumerate(info[List[iii]]['sentences']):
            if len(caption.split(' ')) < 35:

                g_left,g_right = info[List[iii]]['timestamps'][capidx]
                if g_left == -100 or g_right == -100:
                    continue

                anchor_input = generate_anchor_params(all_anchor_list,[g_left,g_right])

                video_names_list.append(str(List[iii]))
                video_duration_list.append(info[List[iii]]['duration'])
                video_actual_frames_num_list.append(video_frames_num)
                sentence_list.append(unicodedata.normalize('NFKD', caption).encode('ascii','ignore'))
                ground_interval_list.append(info[List[iii]]['timestamps'][capidx])
                anchor_input_list.append(anchor_input)
                cnt+=1

            if cnt == BATCH_SIZE:
                batch = h5py.File(output_path+'/'+dataset+'/'+dataset+'_'+str(batch_id)+'.h5','w')
                batch['video_name'] = np.array(video_names_list) # batch_size
                batch['video_duration'] = np.array(video_duration_list) # batch_size
                batch['video_actual_frames_num'] = np.array(video_actual_frames_num_list) # batch_size
                batch['sentence'] = np.array(sentence_list) # batch_size
                batch['ground_interval'] = np.array(ground_interval_list) # batch_size x 2
                batch['anchor_input'] = np.array(anchor_input_list)
                cnt = 0
                batch_id += 1
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

driver('val', output_path)
getlist(output_path,'val')