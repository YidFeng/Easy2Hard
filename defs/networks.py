'''
implementations of network architectures,
where : HDC is Baseline with rHDC blocks,
        HDC-edge is JESS-Net without refinement branch,
        HDC-edge-refine is the complete JESS-Net
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = [
    "Baseline",
    "HDC",
    "HDC_edge",
    "HDC_edge_refine",
]
class ResBlock(nn.Module):  # with 3 convs
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x0 = x
        x = self.triple_conv(x) + x0
        return x

class rHDCblock(nn.Module):
    def __init__(self, dim):
        super(rHDCblock, self).__init__()

        self.conv_3x3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(dim)

        self.conv_3x3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(dim)

        self.conv_3x3_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(dim)

    def forward(self, feature_map):

        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(out_3x3_1)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(out_3x3_2)))
        return out_3x3_3 + feature_map


class Descriptor_Res(nn.Module):
    def __init__(self, in_c, dim, num_block):
        super(Descriptor_Res, self).__init__()
        self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
        nets = []
        for i in range(num_block):
            nets.append(ResBlock(dim))
        self.res16 = nn.ModuleList(nets)

    def forward(self, x):
        x = self.conv_in(x)
        for m in self.res16:
            x = m(x)
        return x

class Descriptor_rHDC(nn.Module):
    def __init__(self, in_c, dim, num_block):
        super(Descriptor_rHDC, self).__init__()
        self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
        nets = []
        for i in range(num_block):
            nets.append(rHDCblock(dim))
        self.res16 = nn.ModuleList(nets)

    def forward(self, x):
        x = self.conv_in(x)
        for m in self.res16:
            x = m(x)
        return x

class Baseline(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(Baseline, self).__init__()
        self.descriptor = Descriptor_Res(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.descriptor(x)
        x = self.interpreter(x)
        return x


class HDC(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDC, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.descriptor(x)
        x = self.interpreter(x)
        return x


class HDC_edge(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDC_edge, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features = torch.cat((x, edge), dim=1)
        out = self.interpreter2(input_features)
        return edge, out


class HDC_edge_refine(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDC_edge_refine, self).__init__()
        self.descriptor = Descriptor_rHDC(in_c=in_c, dim=dim, num_block=num_block)
        self.interpreter1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )
        self.interpreter2 = nn.Sequential(
            nn.Conv2d(dim+1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            rHDCblock(dim),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        out1 = self.interpreter2(input_features1)
        input_features2 = torch.cat((input_features1, out1), dim=1)
        out2 = self.refine(input_features2)
        return edge, out1, out2

