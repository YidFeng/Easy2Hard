import torch.nn as nn
import torch.nn.functional as F
import torch



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

class ConvBlock(nn.Module):  # with 3 HDR convs
    def __init__(self, dim):
        super(ConvBlock, self).__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.triple_conv(x)

class HDRblock(nn.Module):
    def __init__(self, dim):
        super(HDRblock, self).__init__()

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
        # out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))
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

class Descriptor_HDR(nn.Module):
    def __init__(self, in_c, dim, num_block):
        super(Descriptor_HDR, self).__init__()
        self.conv_in = nn.Conv2d(in_c, dim, kernel_size=3, padding=1)
        nets = []
        for i in range(num_block):
            nets.append(HDRblock(dim))
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
            nn.Conv2d(dim + 1, dim, kernel_size=3, padding=1),
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


class HDR(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDR, self).__init__()
        self.descriptor = Descriptor_HDR(in_c=in_c, dim=dim, num_block=num_block)
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


class HDR_edge(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDR_edge, self).__init__()
        self.descriptor = Descriptor_HDR(in_c=in_c, dim=dim, num_block=num_block)
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
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
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


class HDR_edge_refine(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDR_edge_refine, self).__init__()
        self.descriptor = Descriptor_HDR(in_c=in_c, dim=dim, num_block=num_block)
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
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim+1+out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
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

class HDR_edge_frefine(nn.Module):
    def __init__(self, in_c, out_c, dim, num_block):  #numblock is for descriptor
        super(HDR_edge_frefine, self).__init__()
        self.descriptor = Descriptor_HDR(in_c=in_c, dim=dim, num_block=num_block)
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
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, kernel_size=3, padding=1)
        )
        self.frefine = nn.Sequential(
            nn.Conv2d(2+2*out_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            HDRblock(dim),
            nn.Conv2d(dim, 2*out_c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.descriptor(x)
        edge = self.interpreter1(x)
        input_features1 = torch.cat((x, edge), dim=1)
        out1 = self.interpreter2(input_features1)
        out1_f = torch.rfft(out1, 2, onesided=False)
        out1_f = torch.transpose(out1_f, 1, 4).squeeze(4)
        edge_f = torch.rfft(edge, 2, onesided=False)
        edge_f = torch.transpose(edge_f, 1, 4).squeeze(4)
        # x_f = torch.cat((x_f[:, :, :, :, 0], x_f[:, :, :, :, 1]), dim=1)
        input_features2 = torch.cat((out1_f, edge_f), dim=1)
        out2 = self.frefine(input_features2)
        out2 = torch.transpose(out2.unsqueeze(4), 1, 4)
        out2 = torch.irfft(out2, 2, onesided=False)
        return edge, out1, out2







