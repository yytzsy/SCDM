import numpy as np
import h5py
import json
import random
import logging


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



def analysis_iou(result, epoch, logging):

    threshold_list = [0.1,0.3,0.5,0.7]
    rank_list = [1,5,10]
    result_dict = {}
    top1_iou = []

    for i in range(len(result)):
        video_name = result[i][0]
        ground_truth_interval = result[i][1]
        sentence = result[i][2]
        predict_list = result[i][3]
        video_duration = result[i][4]
        predict_score_list = result[i][5]

        iou_list = []
        for predict_interval in predict_list:
            iou_list.append(calculate_IOU(ground_truth_interval,predict_interval))
        top1_iou.append(iou_list[0])

        for rank in rank_list:
            for threshold in threshold_list:
                key_str = 'Recall@'+str(rank)+'_iou@'+str(threshold)
                if key_str not in result_dict:
                    result_dict[key_str] = 0

                for jj in range(rank):
                    if iou_list[jj] >= threshold:
                        result_dict[key_str] += 1
                        break

    logging.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    logging.info('epoch '+str(epoch)+': ')
    for key_str in result_dict:
        logging.info(key_str+': '+str(result_dict[key_str]*1.0/len(result)))
    logging.info('mean iou: '+str(np.mean(top1_iou)))
    logging.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

