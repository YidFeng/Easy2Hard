import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from torchvision import transforms


# Dataset consisting of concatenated image pairs (Ground Truth in the left and Observation in the right)
class TrainDataset(Dataset):
    def __init__(self, dir, dir2, patch_size, aug_data): #dir2 for edge maps
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = dir
        self.edge_dir = dir2
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.aug_data = aug_data
        self.file_num = len(self.mat_files)
        # self.transforms = transforms.Compose([
        #     transforms.functional.adjust_contrast(),
        #     transforms.
        # ])

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        key = file_name.split('.')[0]
        key = int(key) % 200
        edgefile_name = str(key) + '.jpg'
        img_file = os.path.join(self.root_dir, file_name)
        edge_file = os.path.join(self.edge_dir, edgefile_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        edge_label = cv2.imread(edge_file)
        edge_label = cv2.cvtColor(edge_label, cv2.COLOR_BGR2GRAY)

        edge_label[edge_label < 100] = 0
        edge_label[edge_label > 100] = 1


        h, w, _ = img_pair.shape

        if self.aug_data:
            O, B, E = self.crop(img_pair,edge_label, aug=True)
            O, B, E = self.flip(O, B, E)
            O, B, E = self.rotate(O, B, E)
            O, B = self.ToPIL(O), self.ToPIL(B)
            O, B = self.hue(O, B)
            O, B = self.contrast(O, B)
            O, B = self.bright(O, B)
            O, B = self.ToCvArray(O), self.ToCvArray(B)
        else:
            O, B, E = self.crop(img_pair,edge_label, aug=False)

        E = np.expand_dims(E, axis=2)
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        E = np.transpose(E, (2, 0, 1))
        sample = {'OB': O, 'GT': B, 'EG': E}

        return sample

    def crop(self, img_pair, edge, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]
        E = edge[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))
            E = cv2.resize(E, (patch_size, patch_size))

        return O, B, E

    def bright(self, O, B):
        brightness_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_brightness(img=O, brightness_factor=brightness_factor)
        B = TF.adjust_brightness(img=B, brightness_factor=brightness_factor)
        return O, B

    def contrast(self, O, B):
        contrast_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_contrast(img=O, contrast_factor=contrast_factor)
        B = TF.adjust_contrast(img=B, contrast_factor=contrast_factor)
        return O, B

    def hue(self, O, B):
        hue_factor = self.rand_state.uniform(-0.3, 0.3)
        O = TF.adjust_hue(img=O, hue_factor=hue_factor)
        B = TF.adjust_hue(img=B, hue_factor=hue_factor)
        return O, B

    def ToPIL(self, img):
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def ToCvArray(self, img):
        img = np.asarray(img)
        return (img/255.0).astype(np.float32)
    def flip(self, O, B, E):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
            E = np.flip(E, axis=1)
        return O, B, E

    def rotate(self, O, B, E):
        angle = self.rand_state.randint(-45, 45)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        E = cv2.warpAffine(E, M, (patch_size, patch_size))
        return O, B, E


class TestDataset(Dataset):
    def __init__(self, dir, dir2, patch_size):
        super().__init__()
        self.rand_state = RandomState(66)
        self.edge_dir = dir2
        self.root_dir = dir
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        key = file_name.split('.')[0]
        key = int(key) % 200
        edgefile_name = str(key) + '.jpg'
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        edge_file = os.path.join(self.edge_dir, edgefile_name)
        edge_label = cv2.imread(edge_file)
        edge_label = cv2.cvtColor(edge_label, cv2.COLOR_BGR2GRAY)

        edge_label[edge_label < 100] = 0
        edge_label[edge_label > 100] = 1
        O, B, E = self.crop(img_pair, edge_label)
        E = np.expand_dims(E, axis=2)
        B = np.transpose(B, (2, 0, 1))
        O = np.transpose(O, (2, 0, 1))
        E = np.transpose(E, (2, 0, 1))
        sample = {'OB': O, 'GT': B, 'EG': E}
        return sample

    def crop(self, img_pair, edge):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        p_h, p_w = patch_size, patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r + p_h, c + w: c + p_w + w]
        B = img_pair[r: r + p_h, c: c + p_w]
        E = edge[r: r + p_h, c: c + p_w]
        return O, B, E




if __name__ == '__main__':
    dt = TestDataset(dir='../dataset/train',dir2='../dataset/edgeMap', patch_size=128)
    a = dt[0]
    cv2.imshow('O',np.transpose(a['OB'], (1, 2, 0)))
    cv2.imshow('B',np.transpose(a['GT'], (1, 2, 0)))
    cv2.imshow('E',np.transpose(a['EG'], (1, 2, 0))*255)
    cv2.waitKey(0)
    # cv2.imwrite("E:\\Desktop\\OB.png", np.transpose(a['OB']*255, (1, 2, 0)))
    # cv2.imwrite("E:\\Desktop\\GT.png", np.transpose(a['GT']*255, (1, 2, 0)))

    # dt = TestDataset('val')
    # print('TrainValDataset')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
    #
    # print()
    # dt = TestDataset('test')
    # print('TestDataset')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
    #
    # print()
    # print('ShowDataset')
    # dt = ShowDataset('test')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
