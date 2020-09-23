import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import cv2
import argparse
import numpy as np
import logging
import time
import torch
from torch import nn
from torch.nn import MSELoss, L1Loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Myloss import TVLoss, edgeLoss, mL1Loss, GDLoss, mSSIMLoss
from dataset import TrainDataset, TestDataset
import pytorch_ssim

from networks import Baseline, HDR, HDR_edge_refine


def get_args():
    parser = argparse.ArgumentParser(description="train derain model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--log_dir", type=str, default='../logdir',
                        help="log_dir")
    parser.add_argument("--model_dir", type=str, default='../models',
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=128,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")

    parser.add_argument("--num_workers", type=int, default=8,
                        help="numworks in dataloader")
    parser.add_argument("--aug_data", type=bool, default=True,
                        help="whether to augment data")
    parser.add_argument("--lr", type=float, default=0.001 ,
                        help="learning rate")
    parser.add_argument("--loss", type=str, default="MSE",
                        help="loss; MSE', 'L1Loss', or 'MyLoss' is expected")

    parser.add_argument("--checkpoint", type=str, default="the_end",
                        help="model architecture ('Similarity')")
    #############
    parser.add_argument("--net", type=str, default="HDR_edge_refine",
                        help="Baseline, HDR, HDR_edge, HDR_edge_refine is expected")
    parser.add_argument("--opt", type=str, default="Adam",
                        help="Optimizer for updating the network parameters")
    parser.add_argument("--epochs", type=int, default=120,
                        help="number of epochs")
    parser.add_argument("--sessname", type=str, default="HDR_edge_refine",
                        help="different session names for parameter modification")
    parser.add_argument("--train_dir", type=str, default='../dataset/train',
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, default='../dataset/val',
                        help="test image dir")
    parser.add_argument("--edge_dir", type=str, default='../dataset/edgeMap',
                        help="edge label dir")

    args = parser.parse_args()

    return args


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self, args):
        self.log_dir = args.log_dir
        self.model_dir = args.model_dir
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        if args.net == "Baseline":
            self.net = Baseline(in_c=3, out_c=3, dim=64, num_block=18).cuda()
        elif args.net == "HDR":
            self.net = HDR(in_c=3, out_c=3, dim=64, num_block=18).cuda()
        elif args.net == "HDR_edge_refine":
            self.net = HDR_edge_refine(in_c=3, out_c=3, dim=64, num_block=18).cuda()
        elif args.net == "HDR_edge":
            self.net = HDR_edge(in_c=3, out_c=3, dim=64, num_block=18).cuda()
        else:
            print("NET name error!!")
        self.ssim = pytorch_ssim.SSIM().cuda()
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.step = 0
        self.epoch = args.epochs
        self.now_epoch = 0
        self.start_epoch = 0
        self.writers = {}
        self.total_step = 0

        self.sessname = args.sessname
        self.mse = MSELoss().cuda()
        self.l2Loss = MSELoss().cuda()
        self.tvLoss = TVLoss().cuda()
        self.GDLoss = GDLoss().cuda()
        self.l1Loss = L1Loss().cuda()
        self.edgeLoss = edgeLoss().cuda()
        self.mssimloss = mSSIMLoss().cuda()
        if args.opt == "SGD":
            self.opt = SGD(self.net.parameters(), lr=args.lr)
        else:
            self.opt = Adam(self.net.parameters(), lr=args.lr)

        self.sche = MultiStepLR(self.opt, milestones=[30, 60, 90], gamma=0.5)

    def tensorboard(self, name):
        path = os.path.join(self.log_dir, self.sessname)
        ensure_dir(path)
        self.writers[name] = SummaryWriter(os.path.join(path, name+'.events'))
        return self.writers[name]

    def write(self, name, loss, lossL, lossS, lossT, ssim, epoch, image_last_train, image_val):
        lr = self.opt.param_groups[0]['lr']
        self.writers[name].add_scalar("lr", lr, epoch)
        self.writers[name].add_scalars("loss", {"train": loss[0], "test": loss[1]}, epoch)
        self.writers[name].add_scalars("loss_components", {"lossL_train": lossL[0], "lossL_test": lossL[1],
                                                           "lossS_train": lossS[0], "lossS_test": lossS[1],
                                                           "lossT_train": lossT[0], "lossT_test": lossT[1]}, epoch)
        self.writers[name].add_scalars("ssim", {"train": ssim[0], "test": ssim[1]}, epoch)


    def write_close(self,name):
        self.writers[name].close()

    def get_dataloader(self, dir, dir2, name):
        if name == "train":
            dataset = TrainDataset(dir, dir2, self.image_size, aug_data=args.aug_data)
            a = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, drop_last=True)
            self.total_step = len(a)
            return a
        elif name == "val":
            dataset = TestDataset(dir, dir2, self.image_size)
            return DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)
        else:
            print("Incorrect Name for Dataloader!!!")
            return 0

    def save_checkpoints(self, name):
        dir = os.path.join(self.model_dir, self.sessname)
        ensure_dir(dir)
        ckp_path = os.path.join(dir, name)
        obj = {
            'net': self.net.state_dict(),
            'now_epoch': self.now_epoch+1,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, dir):
        ckp_path = dir
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.start_epoch = obj['now_epoch']

    def inf_batch(self, name, batch):
        OB, GT, EG = batch['OB'].cuda(), batch['GT'].cuda(), batch['EG'].cuda()
        edge, pre, res = self.net(OB)
        ####
        lossL = self.l1Loss(pre, GT) + self.l1Loss(pre+res, GT)
        lossS = 4 - self.mssim(pre, GT) + 4 - self.mssim(pre+res, GT)
        lossD = self.GDLoss(pre, mask=1-EG) + self.GDLoss(pre+res, mask=1-EG)
        lossT = self.edgeLoss(edge, EG)
        loss = lossL+4*lossT
        ########
        ssim = self.ssim(pre+res, GT)
        psnr = 10*torch.log10((1.0/self.mse(pre+res, GT)))
        if name == 'train':
            self.net.zero_grad()
            loss.backward()
            self.opt.step()
            lr_now = self.opt.param_groups[0]["lr"]
            logger.info("epoch %d/%d: step %d/%d: loss is %f ssim is %f psnr is %f lr is %f"
                        % (self.now_epoch, self.epoch, self.step, self.total_step, loss, ssim, psnr,  lr_now))
            self.step += 1

        return pre+res, loss.item(), lossL.item(), lossS.item(), lossT.item(), ssim.item(), psnr.item()

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (6, 2)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)

    def epoch_out(self):
        self.step = 0


def run_train_val(args):

    sess = Session(args)
    # sess.load_checkpoints("E:\Desktop\\text_choose_net\models\SKNet24_SGD0.01_500e_2branch_L1+SSIM_aug\epoch 158_ssim 0.857511")
    sess.tensorboard('JESS5000')
    ssim_m = 0.0
    sess.now_epoch = sess.start_epoch
    for epoch in range(int(sess.epoch-sess.start_epoch)):
        start_time = time.time()
        epoch = epoch + sess.start_epoch
        dt_train = sess.get_dataloader(dir=args.train_dir, dir2=args.edge_dir, name='train')
        dt_val = sess.get_dataloader(dir=args.test_dir, dir2=args.edge_dir, name='val')
        sess.net.train()
        loss_train = []
        lossL_train = []
        lossS_train = []
        lossT_train = []
        ssim_train = []
        psnr_train = []
        for batch in dt_train:
            result_train, loss, lossL, lossS, lossT, ssim, psnr = sess.inf_batch("train", batch)
            loss_train.append(loss)
            lossL_train.append(lossL)
            lossS_train.append(lossS)
            lossT_train.append(lossT)
            ssim_train.append(ssim)
            psnr_train.append(psnr)
        sess.epoch_out()
        loss_test = []
        lossL_test = []
        lossS_test = []
        lossT_test = []
        ssim_test = []
        psnr_test = []
        sess.net.eval()
        with torch.no_grad():
            for batch in dt_val:
                result_val, loss, lossL, lossS, lossT, ssim, psnr = sess.inf_batch("val", batch)
                loss_test.append(loss)
                lossL_test.append(lossL)
                lossS_test.append(lossS)
                lossT_test.append(lossT)
                ssim_test.append(ssim)
                psnr_test.append(psnr)
            sess.write(name="JESS5000", loss=[np.mean(loss_train), np.mean(loss_test)], lossL=[np.mean(lossL_train), np.mean(lossL_test)],
                       lossS=[np.mean(lossS_train), np.mean(lossS_test)],lossT=[np.mean(lossT_train), np.mean(lossT_test)],
                       ssim=[np.mean(ssim_train), np.mean(ssim_test)], epoch=epoch, image_last_train=result_train, image_val=result_val)
            ssim_now = np.mean(ssim_test)
            psnr_now = np.mean(psnr_test)
            if ssim_now > ssim_m:
                logger.info('ssim increase from %f to %f now, psnr %f' % (ssim_m, ssim_now, psnr_now))
                ssim_m = ssim_now
                sess.save_checkpoints("epoch %d_ssim %f_psnr %f " % (epoch, ssim_m,psnr_now))
                logger.info('save model as epoch_%d_ssim %f_psnr %f' % (epoch, ssim_m, psnr_now))
            elif epoch % 30 == 0 :
                sess.save_checkpoints("epoch %d_ssim %f_psnr %f " % (epoch, ssim_now, psnr_now))
                logger.info('save model as epoch_%d_ssim %f_psnr %f' % (epoch, ssim_now, psnr_now))
            else:
                logger.info("ssim now is %f, not increase from %f" % (ssim_now, ssim_m))
        sess.now_epoch += 1
        sess.sche.step(epoch=epoch)
        end_time = time.time()
        logger.info("this epoch costs time: %f" % (end_time - start_time))
    sess.write_close("JESS5000")





if __name__ == '__main__':
    log_level = 'info'
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = get_args()
    run_train_val(args=args)

