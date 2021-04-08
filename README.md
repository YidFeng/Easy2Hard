### Easy2Hard: Learning to Handle the Intractables from a Synthetic Dataset for Structure-preserving Image Smoothing
Created by Yidan Feng, from Nanjing University of Aeronautics and Astronautics.

### Introduction
![](intro.png)
This code is implemented with Python 3.6.7, Pytorch 1.3.1 and CUDA 10.1.

The dataset can be downloaded from (to be updated after publication), which should be put into the 'dataset' folder.

The following sources can be downloaded fron Google drive:

The trained model (of JESS-Net): https://drive.google.com/file/d/1R8Fg6MEf99gTMheIgshIcz1lBnzrvzyD/view?usp=sharing

SPS ground-truth files: https://drive.google.com/drive/folders/1-nzVGifUfbufTvDB-zfYAU9XRaNWrH3C?usp=sharing

and texture patternes: https://drive.google.com/drive/folders/1OSyBNEuPmDTTWauw9qUEuU1KdT5odF_Y?usp=sharing

Datasets for ground-truth ablation study: https://drive.google.com/drive/folders/18LxdOG06l83W3OgiM0wu8c-nDsONY-oJ?usp=sharing

The trained models for ground-truth ablation study: https://drive.google.com/drive/folders/1EfgX-VDMOyuqRfJfjq7p11O4rKekXdaY?usp=sharing

To train from sratch:
```bash
python train.py --sessname user_defined --net HDR_edge_refine
````
To show results of real images with a train_model
```bash
python show.py 
