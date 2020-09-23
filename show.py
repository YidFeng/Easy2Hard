import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import h5py
import os
import numpy as np
from train import ensure_dir
import math
import time
import torch.nn as nn
from networks import Baseline, HDR, HDR_edge, HDR_edge_refine

modelPath = ".\\trained_model\epoch 224_ssim 0.922825_psnr 31.733277"

TEST = ["temp"] # a list of the names of test folders

SESSNAME = "OurBest" # will create the current folder named SESSNAME
NET = "HDR_edge_refine"  # "HDR" "HDR_feaFus"

def cal_psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(1.0**2/mse)

def get_image(image):
    image = image*[255]
    # image = 100 + image*5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def load_checkpoints(dir):
    ckp_path = dir
    try:
        obj = torch.load(ckp_path)
        print('Load checkpoint %s' % ckp_path)
        return obj
    except FileNotFoundError:
        print('No checkpoint %s!!' % ckp_path)
        return

def run_test():
    with torch.no_grad():
        if NET == "Baseline":
            net = Baseline(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif NET == "HDR":
            net = HDR(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif NET == "HDR_feaFus":
            net = HDR_feaFus(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif NET == "HDR_edge":
            net = HDR_edge(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif NET == "HDR_edge_refine":
            net = HDR_edge_refine(in_c=3, out_c=3, dim=64, num_block=20).cuda()
        else:
            print("NET name error!!")
        net.eval()
        obj = load_checkpoints(modelPath)
        net.load_state_dict(obj['net'])

        image_files = list(Path(input_dir).glob("*.*"))

        for image_file in image_files:
            # start_time = time.time()
            image_name = str(image_file).split("\\")[-1]

            image_o = (cv2.imread(str(image_file))/255.0).astype(np.float32)
            h, w, c = image_o.shape

            image_o = np.transpose(image_o, (2, 0, 1))
            image_o = torch.from_numpy(np.expand_dims(image_o, axis=0)).type(torch.FloatTensor).cuda()

            if NET == "HDR_edge":
                edge, result = net(image_o)
            elif NET == "HDR_edge_refine":
                edge, result, res = net(image_o)
                result = result+res
            else:
                result = net(image_o)
            result = result.cpu().detach().numpy()
            result = np.transpose(result[0], (1, 2, 0))
            result = get_image(result)
            cv2.imwrite(outout_dir + "/%s" % image_name, result)
            # end_time = time.time()

if __name__ == '__main__':
    for i in TEST:
        input_dir = "../dataset/test/" + i
        subdir = os.path.join(SESSNAME, "show")
        outout_dir = os.path.join("../result", subdir)
        outout_dir = outout_dir + "/" + i
        ensure_dir(outout_dir)
        run_test()
