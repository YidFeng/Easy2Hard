import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.nn.functional import max_pool2d
from pytorch_ssim import SSIM

class edgeLoss(nn.Module):
    def __init__(self):
        super(edgeLoss, self).__init__()
    def forward(self, prediction, label):
        label = label.long()
        mask = (label != 0).float()
        num_positive = torch.sum(mask).float()
        num_negative = mask.numel() - num_positive
        # print (num_positive, num_negative)
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)
        cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(), label.float(), weight=mask, reduce=False)
        return torch.sum(cost)

class mL1Loss(nn.Module):
    def __init__(self):
        super(mL1Loss, self).__init__()
        self.L_0 = nn.L1Loss()
        self.L_1 = nn.L1Loss()
        self.L_2 = nn.L1Loss()
        self.L_3 = nn.L1Loss()
    def forward(self, prediction, label):
        p1, l1 = max_pool2d(input=prediction, kernel_size=2, stride=2), max_pool2d(input=label, kernel_size=2, stride=2)
        p2, l2 = max_pool2d(input=p1, kernel_size=2, stride=2), max_pool2d(input=l1, kernel_size=2, stride=2)
        p3, l3 = max_pool2d(input=p2, kernel_size=2, stride=2), max_pool2d(input=l2, kernel_size=2, stride=2)
        return self.L_0(prediction, label) + self.L_1(p1, l1) + self.L_2(p2, l2) + self.L_3(p3, l3)

class mSSIMLoss(nn.Module):
    def __init__(self):
        super(mSSIMLoss, self).__init__()
        self.L_0 = SSIM()
        self.L_1 = SSIM()
        self.L_2 = SSIM()
        self.L_3 = SSIM()
    def forward(self, prediction, label):
        p1, l1 = max_pool2d(input=prediction, kernel_size=2, stride=2), max_pool2d(input=label, kernel_size=2, stride=2)
        p2, l2 = max_pool2d(input=p1, kernel_size=2, stride=2), max_pool2d(input=l1, kernel_size=2, stride=2)
        p3, l3 = max_pool2d(input=p2, kernel_size=2, stride=2), max_pool2d(input=l2, kernel_size=2, stride=2)
        return self.L_0(prediction, label) + self.L_1(p1, l1) + self.L_2(p2, l2) + self.L_3(p3, l3)

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class GDLoss(nn.Module):
    def __init__(self):
        super(GDLoss, self).__init__()

    def forward(self,x,mask):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        w_g = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        w_e = x[:, :, :, 0] - x[:, :, :, w_x-1]
        w_e = torch.unsqueeze(w_e, 3)
        w_g1 = torch.cat((w_g, w_e),3)
        h_g = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        h_e = x[:, :, 0, :] - x[:, :, h_x - 1, :]
        h_e = torch.unsqueeze(h_e, 2)
        h_g1 = torch.cat((h_g, h_e), 2)
        h_tv = (mask * torch.pow(h_g1, 2)).sum()
        w_tv = (mask * torch.pow(w_g1, 2)).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

if __name__ == "__main__":
    a = torch.randn((1, 3, 128, 128)).cuda()
    b = torch.randn((1, 3, 128, 128)).cuda()

    loss = GDLoss().cuda()
    c = loss(a)

    print(b)
    # print(c)

