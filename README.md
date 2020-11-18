# 3D Human Pose Estimation based on Multi-Input Multi-Output Convolutional Neural Network and Event Cameras: a proof of concept on the DHP19 dataset

Human pose estimation is one of the main topics in Computer vision and Deep Learning fields. This repository describes the process of estimating human’s limbs positions using data from event cameras.

***Dataset***

The dataset used is Dynamic Vision Sensor (DVS) 3D Human Pose Dataset, in which 4 synchronized DVS cameras are used to record 33 specific movements from 17 different subjects while the Vicon motion capture system is used to generate position markers in 3D space in order to get groundtruth. More information and descriptions are available on the website https://sites.google.com/view/dhp19/home, with a section for download.

***Data preprocessing***

To preprocess DVS and Vicon data were followed steps described and implemented in https://github.com/SensorsINI/DHP19/tree/master/generate_DHP19 in which DVS frames are generated by accumulating a fixed number of events. Label positions were generated knowing the initial and final event timestamps for the DVS-frame and calculating average position in that time window.

## Training & Testing

***SISO architecture***

The approach used is the same described in *E. Calabrese, G. Taverni, C. Awai Easthope, S. Skriabine, F. Corradi, L. Longinotti, K. Eng, and T. Delbruck, “DHP19: Dynamic vision sensor 3D human pose dataset,” in IEEE Conf. Comput. Vis. Pattern Recog. Workshops (CVPRW), 2019*.

Training can be executed, after executing *file_generation_singleview.py*, through *single_input_training.py*. 

***MIMO architecture***

This is the proposed architecture, described in detail in paper.pdf, which process 2 frames simultaneously making use of shared layers.

Training can be executed through *multi_input_training.py*.

For testing purpouses use *testing_with_conf.py* in which can be set a confidence threshold on predicted positions probability; eventually set confidence parameter to 0 to avoid using confidence threshold mechanism.

More detailed info can ben found in paper.pdf.
