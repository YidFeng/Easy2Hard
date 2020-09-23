### Easy2Hard
The implementation of paper "Easy2Hard: Learning to Handle the Intractables from a Synthetic Dataset for Structure-preserving Image Smoothing". Created by Feng Yidan, from Nanjing University of Aeronautics and Astronautics.

### Instructions
This code is implemented with Python 3.6.7, Pytorch 1.3.1 and CUDA 10.1.

The dataset can be downloaded from (to be updated after publication), which should be put into the 'dataset' folder.

The trained model can be downloaded from (to be updated after publication), which should be put into the 'trained_model' folder.

To train from sratch:
```bash
python train.py --sessname user_defined --net HDR_edge_refine
````
To show results of real images with a train_model
```bash
python show.py 
