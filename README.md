# SCDM
Code for the paper: "Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos"

## requirements
* python 2.7
* tensorflow 1.14.0
* keras 1.2.1


## Introduction
 
Temporal sentence grounding (TSG) in videos aims to detect and localize one target video segment, which semantically corresponds to a given sentence query. We propose a semantic conditioned dynamic modulation (SCDM) mechanism to help solve the TSG problem, which relies on the sentence semantics to modulate the temporal convolution operations for better correlating and composing the sentence-related video contents over time.

![](https://github.com/yytzsy/SCDM/blob/master/task.PNG)

## Download Features and Example Preprocessed Data

First, download the following files into the '**./data**' folder:
* Extracted video features: [charades_i3d_rgb.hdf5](https://cloud.tsinghua.edu.cn/f/fd155bd64baf4359ada9/?dl=1), [activitynet_c3d_fc6_stride_1s.hdf5](https://cloud.tsinghua.edu.cn/d/203f17ed08ae41f69a0e/) (The video feature file is too big to upload and we have divided it into 10 parts, and you should download and merge the 10 parts into a whole feature file), [tacos_c3d_fc6_nonoverlap.hdf5](https://cloud.tsinghua.edu.cn/f/876b5550bd884bf1bb77/?dl=1)
* For glove word embeddings used in our work, please download [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip), and preprocess the word embedding .txt file to a glove.840B.300d_dict.npy file, making it a dict whose key is a word and the corresponding value is the 300-d word embedding. 

## Data Preprocessing

As denoted in our paper, we perform the temporal sentence grounding task in three datasets: Charades-STA, ActivityNet Captions, and TACoS. Before the model training and testing in these three datasets, please preprocess the data first. 

* Go to the '**./grounding/Charades-STA/data_preparation/**' folder, and run:
```
python generate_charades_data.py
```
Preprocessed data will be put into the './data/Charades/h5py/' folder.

* Go to the '**./grounding/TACOS/data_preparation/**' folder, and run:
```
python generate_tacos_data.py
```
Preprocessed data for the TACoS dataset will be put into the './data/TACOS/h5py/' folder.

* Go to the '**./grounding/ActivityNet/data_preparation/**' folder, and run:
```
python generate_anet_data.py
```
Preprocessed data for the ActivityNet Captions dataset will be put into the './data/ActivityNet/h5py/' folder.

## Model Training and Testing
![](https://github.com/yytzsy/SCDM/blob/master/model.PNG)

* For the Charades-STA dataset, the proposed model and all its variant models are provided. For example, the proposed SCDM model implementation is in the '**./grounding/Charades-STA/src_SCDM**' folder, run:
```
python run_charades_scdm.py --task train
```
for model training, and run:
```
python run_charades_scdm.py --task test
```
for model testing. Other variant models are similar to train and test.

* For the TACoS and ActivityNet Captions dataset, we only provide the proposed SCDM model implementation in the '**./grounding/xxx/src_SCDM**' folder. The training and testing process are similar to the Charades-STA dataset.
* Please train our provided models from scratch, and you can reproduce the results in the paper (not exactly the same, but almost).

## Citation
```
@inproceedings{yuan2019semantic,
  title={Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos},
  author={Yuan, Yitian and Ma, Lin and Wang, Jingwen and Liu, Wei and Zhu, Wenwu},
  booktitle={Advances in Neural Information Processing Systems},
  pages={534--544},
  year={2019}
}
```
