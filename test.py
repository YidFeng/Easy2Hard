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
from networks import Baseline, HDR, HDR_feaFus
import pytorch_ssim
import csv
import shutil



#生成csv文件，包括单张SSIM/PSNR，平均SSIM/PSNR；保存低于平均指标的文件列表；可以选择输出高于平均的图片，或低于平均的图片；

modelPath = "C:\dan_temp\\text_code\models\HDR_L1_12D\epoch 110_ssim 0.901637_psnr 30.682475"
TEST = ["train100"] #"train"
SESSNAME = "HDR_L1_12D"
NET = "HDR"  # "HDR" "HDR_feaFus"

def cal_psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(1.0**2/mse)

def get_image(image):
    image = image*[255]
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

def predict(img, net, gt):
    SSIM = pytorch_ssim.SSIM().cuda()
    image_o = np.transpose(img, (2, 0, 1))
    image_o = torch.from_numpy(np.expand_dims(image_o, axis=0)).type(torch.FloatTensor).cuda()
    image_g = np.transpose(gt, (2, 0, 1))
    image_g = torch.from_numpy(np.expand_dims(image_g, axis=0)).type(torch.FloatTensor).cuda()
    result = net(image_o)
    ssim = SSIM(result, image_g)
    result = result.cpu().detach().numpy()
    result = np.transpose(result[0], (1, 2, 0))
    return result, ssim.item()

def run_test(outdir):
    with torch.no_grad():
        if NET == "Baseline":
            net = Baseline(in_c=3, out_c=3, dim=64, num_block=8).cuda()
        elif NET == "HDR":
            net = HDR(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif NET == "HDR_feaFus":
            net = HDR_feaFus(in_c=3, out_c=3, dim=64, num_block=8).cuda()
        else:
            print("NET name error!!")
        net.eval()
        obj = load_checkpoints(modelPath)
        net.load_state_dict(obj['net'])
        psnr_list = []
        ssim_list = []
        name_list = []

        image_files = list(Path(input_dir).glob("*.*"))
        f = open(os.path.join(outdir,"info.csv"), 'w')
        f_csv = csv.writer(f)
        f_csv.writerow(["IMAGE", "SSIM", "PSNR"])
        for image_file in image_files:
            start_time = time.time()
            image_name = str(image_file).split("\\")[-1]
            name_list.append(image_name)
            image_pair = (cv2.imread(str(image_file))/255.0).astype(np.float32)
            h, ww, c = image_pair.shape
            w = ww//2
            image_g = image_pair[:, :w, :]
            image_o = image_pair[:, w:, :]
            result, ssim = predict(img=image_o, net=net, gt=image_g)
            psnr = cal_psnr(result, image_g)
            ssim_list.append(ssim)
            psnr_list.append(psnr)
            f_csv.writerow([image_name, ssim, psnr])
            # result = result.cpu().detach().numpy()
            # result = np.transpose(result[0], (1, 2, 0))
            result = get_image(result)
            cv2.imwrite(outout_dir + "/%s" % image_name, result)
            end_time = time.time()
            print("Process %s, time %f"% (image_name, (end_time - start_time)))
        ssim_list = np.array(ssim_list)
        psnr_list = np.array(psnr_list)
        ssim_mean = np.mean(ssim_list)
        psnr_mean = np.mean(psnr_list)
        f_csv.writerow(['ssim_mean', 'psnr_mean'])
        f_csv.writerow([ssim_mean, psnr_mean])
        f.close()

        bad_index = np.argwhere(ssim_list < ssim_mean)
        good_index = np.argwhere(ssim_list > ssim_mean)
        bad_images = np.array(name_list)[bad_index].flatten()
        good_images = np.array(name_list)[good_index].flatten()
        np.save(os.path.join(outdir, "bad.npy"), bad_images)
        np.save(os.path.join(outdir, "good.npy"), good_images)

def vis_npy(file):
    vis_dir = os.path.join(outout_dir,file.split('.')[0])
    ensure_dir(vis_dir)
    names = np.load(outout_dir + "/%s" % file)
    for n in names:
        shutil.copy(src=outout_dir + "/%s" % n, dst=vis_dir + "/%s" % n)


if __name__ == '__main__':
    for i in TEST:
        input_dir = os.path.join("../dataset", i)
        subdir = os.path.join(SESSNAME, "test")
        outout_dir = os.path.join("../result", subdir)
        outout_dir = outout_dir + "/" + i
        ensure_dir(outout_dir)
        run_test(outout_dir)
        # vis_npy("good.npy")

