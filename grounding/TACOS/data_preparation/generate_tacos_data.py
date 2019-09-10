import numpy as np
import os, json, h5py, math, pdb, glob
import unicodedata
import cPickle as pkl


options = {}
options['feature_map_len']=[256,128,64,32,16]
options['scale_ratios_anchor1']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor2']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor3']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor4']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor5']=[0.25,0.5,0.75,1]


SAMPLE_lEN = 1024
BATCH_SIZE = 16

output_path = '../../../data/TACOS/h5py/'
splitdataset_path = '../../../data/TACOS/datasplit_info/tacos_split.npz'
train_captions_path = '../../../data/TACOS/datasplit_info/train.json'
val_captions_path = '../../../data/TACOS/datasplit_info/val.json'
test_captions_path = '../../../data/TACOS/datasplit_info/test.json'

train_j = json.load(open(train_captions_path))
val_j = json.load(open(val_captions_path))
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



def get_ground_truth_position(ground_position):
    left_frames = ground_position[0]
    right_frames = ground_position[1]
    left_position = int(left_frames / 29.4)
    right_position = int(right_frames / 29.4)
    if left_position < 0 or right_position < left_position:
        return -1,-1
    else:
        return left_position,right_position


def getlist(output_path, split):
    List = glob.glob(output_path+'/'+split+'/'+'*.h5')
    f = open(output_path+'/'+split+'/'+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')



def get_video_data_HL(video_data_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)



def driver(dataset, output_path):
    if dataset == 'train':
        info = train_j
    elif dataset == 'val':
        info = val_j
    elif dataset == 'test':
        info = test_j

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
    List = np.load(splitdataset_path)[dataset] # get the train,val or test training video name
    for iii in range(len(List)):
        video_name =  List[iii]
        for capidx, caption in enumerate(info[List[iii]]['sentences']):
            if len(caption.split(' ')) < 35:

                g_left,g_right = get_ground_truth_position(info[List[iii]]['timestamps'][capidx])
                if g_left == -1 or g_right == -1:
                    continue

                g_position = [g_left,g_right]
                anchor_input = generate_anchor_params(all_anchor_list,g_position)

                video_names_list.append(str(List[iii]))
                video_duration_list.append(info[List[iii]]['duration'])
                sentence_list.append(unicodedata.normalize('NFKD', caption).encode('ascii','ignore'))
                ground_interval_list.append(info[List[iii]]['timestamps'][capidx])
                anchor_input_list.append(anchor_input)
                cnt+=1

            if cnt == BATCH_SIZE:
                batch = h5py.File(output_path+'/'+dataset+'/'+dataset+'_'+str(batch_id)+'.h5','w')
                batch['video_name'] = np.array(video_names_list) # batch_size
                batch['video_duration'] = np.array(video_duration_list) # batch_size
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



def shuffle_train_data():

    video_data_path_train = output_path+'train/train.txt'
    new_path = output_path+'shuffle_train/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)


    video_list_train = get_video_data_HL(video_data_path_train)
    h5py_part_list = [[] for i in range(100)]
    for i in range(len(video_list_train)):
        index = i % 100
        h5py_part_list[index].append(video_list_train[i])

    count = 1
    for part_list in h5py_part_list:
        fname = []
        title = []
        anchor_input = np.zeros([BATCH_SIZE*len(part_list),5,256,12])+0.0
        timestamps = []
        duration = []
        for idx, item in enumerate(part_list):
            print item
            current_batch = h5py.File(item,'r')
            current_fname = current_batch['video_name']
            current_title = current_batch['sentence']
            current_timestamps = current_batch['ground_interval']
            current_duration = current_batch['video_duration']
            current_anchor_input = current_batch['anchor_input']
            fname = fname + list(current_fname)
            title = title + list(current_title)
            timestamps = timestamps + list(current_timestamps)
            duration = duration + list(current_duration)
            anchor_input[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:,:] = current_anchor_input
        index = np.arange(BATCH_SIZE*len(part_list))
        np.random.shuffle(index)
        fname = [fname[i] for i in index]
        title =  [title[i] for i in index]
        timestamps = [timestamps[i] for i in index]
        duration = [duration[i] for i in index]
        anchor_input = anchor_input[index,:,:,:]
        for idx,item in enumerate(part_list):
            batch = h5py.File(new_path+'train'+str(count)+'.h5','w')
            batch['video_name'] = fname[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch['sentence'] = title[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch['ground_interval'] = timestamps[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch['video_duration'] = duration[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch['anchor_input'] = anchor_input[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:,:]
            count = count + 1

    List = glob.glob(new_path+'*.h5')
    f = open(new_path+'train.txt','w')
    for ele in List:
        f.write(ele+'\n')



driver('train', output_path)
getlist(output_path,'train')

driver('test', output_path)
getlist(output_path,'test')

shuffle_train_data()

