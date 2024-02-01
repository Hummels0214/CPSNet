import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.dca import DCA


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(5 * channel, 5 * channel, 3, padding=1)
        self.conv_change = BasicConv2d(channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(5 * channel, channel, 3, padding=1)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.transconv = BasicConv2d(2 * channel, channel, 3, padding=1)

    #
    def forward(self, x1, x2, x3, t1, t2):
        x1_0 = x1  # 32,11,11

        x2_0 = x2  # 32,11,11#32,22,22

        x3_0 = x3  # 32,11,11#32,44,44
        x2_1 = self.conv_upsample1(self.upsample(x1_0)) * x2_0  # 32,22,22
        x2_1 = torch.cat((t2, self.conv_upsample4(self.upsample(x1_0))), dim=1)  # 64,22,22
        x3_1 = self.conv_upsample1(self.upsample(x2_0)) * x3_0  # 32,44,44
        x3_1 = torch.cat((t1, self.conv_upsample4(self.upsample(x2))), dim=1)  # 64,44,44
        x3_2 = self.conv_upsample5(self.upsample(x2_1)) * x3_1 * self.conv_change(x3_0)  # 64,44,44
        x3_2 = self.conv_concat4(
            torch.cat((self.conv_concat3(torch.cat((x3_2, self.conv_upsample5(self.upsample(x2_1))), dim=1)), x3_0),
                      dim=1))  # 160,44,44
        x3_2 = self.conv4(x3_2)  # 32,44,44
        x3_1 = self.transconv(x3_1)  # 32,44,44
        # x2_1 = self.transconv(self.upsample(x2_1))

        # ------------------------------------------------------------------------------------------------------------------------------------------------------
        # x2_1 = self.conv_upsample1(self.upsample(x1_0)) * x2_0#32,22,22
        # x2_1 = torch.cat((x2_1,self.conv_upsample4(self.upsample(x1_0))),dim=1)#64,44,44
        # x3_1 = self.conv_upsample1(self.upsample(x2_0)) * x3_0#32,44,44
        # x3_1 = torch.cat((x3_1,self.conv_upsample4(self.upsample(x2_0))),dim=1)#64,44,44
        # x3_2 = self.conv_upsample5(self.upsample(x2_1)) * x3_1 * self.conv_change(x3_0)#64,44,44
        # x3_2 = self.conv_concat4(torch.cat((self.conv_concat3(torch.cat((x3_2,self.conv_upsample5(self.upsample(x2_1))),dim=1)),x3_0),dim=1))#160,44,44
        # x3_2 = self.conv4(x3_2)#32,44,44
        # x3_1 = self.transconv(x3_1)#32,44,44
        # ------------------------------------------------------------------------------------------------------------------------------------------------------
        return x3_2, x3_1



class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class TEM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(TEM, self).__init__()
        self.normalize = normalize

        self.num_s = int(plane_mid)
        self.num_in = num_in
        self.num_n = (mids) * (mids)  # 16
        self.conv_fc = nn.Conv2d(self.num_in * 2, self.num_s, kernel_size=1)
        self.conv_out = nn.Conv2d(self.num_in * 2, self.num_in, kernel_size=1)
        # f1
        self.conv_f1_Q = nn.Conv2d(self.num_in, self.num_s, kernel_size=1)
        self.conv_f1_K = nn.Conv2d(self.num_in, self.num_s, kernel_size=1)
        self.ap_f1 = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.gcn_f1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_f1_extend = nn.Conv2d(self.num_s, self.num_in, kernel_size=1, bias=False)

        # f2
        self.conv_f2_Q = nn.Conv2d(self.num_in, self.num_s, kernel_size=1)
        self.conv_f2_K = nn.Conv2d(self.num_in, self.num_s, kernel_size=1)
        self.ap_f2 = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.gcn_f2 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_f2_extend = nn.Conv2d(self.num_s, self.num_in, kernel_size=1, bias=False)

    def forward(self, T1, T2):
        T2 = F.upsample(T2, (T1.size()[-2], T1.size()[-1]))  # n*32*44*44
        n, c, h, w = T1.size()
        f1, f2 = T1, T2
        Tc = self.conv_fc(torch.cat((f1, f2), dim=1))  # n*16*44*44
        fc_att = torch.nn.functional.softmax(Tc, dim=1)[:, 1, :, :].unsqueeze(1)  # n*1*44*44
        # --------------------f1------------------------------
        f1_Q = self.conv_f1_Q(f1).view(n, self.num_s, -1).contiguous()  # n*16*1936
        f1_k = self.conv_f1_K(f1)  # n*16*1936
        f1_masked = f1_k * fc_att  # n*16*44*44
        f1_V = self.ap_f1(f1_masked)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)  # n*16*16
        f1_proj_reshaped = torch.matmul(f1_V.permute(0, 2, 1), f1_k.reshape(n, self.num_s, -1))  # n*16*1936
        f1_proj_reshaped = torch.nn.functional.softmax(f1_proj_reshaped, dim=1)  # [n,16,1936] Tv
        f1_rproj_reshaped = f1_proj_reshaped  # [n,16,1936]
        f1_n_state = torch.matmul(f1_Q, f1_proj_reshaped.permute(0, 2, 1))  # before GCN n*16*16

        f1_n_rel = self.gcn_f1(f1_n_state)  # [n,16,16]
        f1_state_reshaped = torch.matmul(f1_n_rel, f1_rproj_reshaped)  # [n,16,1936]
        f1_state = f1_state_reshaped.view(n, self.num_s, *f1.size()[2:])  # [n,16,44,44]
        f1_out = f1 + (self.conv_f1_extend(f1_state))  # [n,32,44,44]
        #         #---------------------f2----------------------------
        f2_Q = self.conv_f2_Q(f2).view(n, self.num_s, -1).contiguous()  # n*16*1936
        f2_k = self.conv_f1_K(f2)  # n*16*1936
        f2_masked = f2_k * fc_att  # n*16*44*44
        f2_V = self.ap_f2(f2_masked)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)  # n*16*16
        f2_proj_reshaped = torch.matmul(f2_V.permute(0, 2, 1), f2_k.reshape(n, self.num_s, -1))  # n*16*1936
        f2_proj_reshaped = torch.nn.functional.softmax(f2_proj_reshaped, dim=1)  # [n,16,1936] Tv
        f2_rproj_reshaped = f2_proj_reshaped  # [n,16,1936]
        f2_n_state = torch.matmul(f2_Q, f2_proj_reshaped.permute(0, 2, 1))  # before GCN n*16*16

        f2_n_rel = self.gcn_f2(f2_n_state)  # [n,16,16]
        f2_state_reshaped = torch.matmul(f2_n_rel, f2_rproj_reshaped)  # [n,16,1936]
        f2_state = f2_state_reshaped.view(n, self.num_s, *f2.size()[2:])  # [n,16,44,44]
        f2_out = f2 + (self.conv_f2_extend(f2_state))  # [n,32,44,44]
        # ---------concat-------------------
        # f_out = self.conv_out(torch.cat(f1_out, f2_out, dim=1))
        f_out = self.conv_out(torch.cat((f1_out, f2_out), dim=1))
        return f_out



class CustomModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            # nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            # nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=6, dilation=3),
            # nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            # nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),

        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=9, dilation=3),
            # nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            # nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),

        )
        self.conv = BasicConv2d(out_channels, out_channels, 3, padding=1)

        self.catconv = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        # self.catconv = BasicConv2d(4 * out_channels, out_channels, 1)
        self.eca = EfficientChannelAttention(64)

    def forward(self, x):
        x1 = self.eca(x) * x
        out1 = self.conv1(x1)
        out2 = self.conv3(x1)
        out3 = self.conv5(x1)
        out4 = self.conv7(x1)
        out1 = torch.cat((out1, out2, out3, out4), dim=1)
        out1 = self.catconv(out1)

        x2 = self.conv(x1)
        out = out1 + x2

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        # 这里可以对照上一版代码，理解每一个函数的作用
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PolypPVT(nn.Module):
    def __init__(self, channel=32,
                 patch_size=4,
                 n=1,

                 ):
        super(PolypPVT, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.tem = TEM()

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)
        self.ctm = CustomModule(64, 64)
        self.DCA = DCA(n=n,
                       features=[int(32), int(32), int(32)],
                       strides=[1, 1, 1],
                       patch=1,
                       spatial_att=True,
                       channel_att=True,
                       spatial_head=[4, 4, 4],
                       channel_head=[1, 1, 1],
                       )
        # self.ctm128 = CustomModule(128,128)
        # self.ctm320 = CustomModule(320, 320)
        # self.ctm512 = CustomModule(512, 512)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        # CIM
        # x1 = self.ca(x1) * x1 # channel attention
        # cim_feature = self.sa(x1) * x1 # spatial attention
        ctm_feature = self.ctm(x1)
        # x2 = self.ctm128(x2)
        # x3 = self.ctm320(x3)
        # x4 = self.ctm320(x4)

        # CFM
        x1_t = self.Translayer2_0(x1)
        x2_t = self.Translayer2_1(x2)  # 32 44 44
        x3_t = self.Translayer3_1(x3)  # 32 22 22
        x4_t = self.Translayer4_1(x4)  # 32 11 11
        x2_d, x3_d, x4_d = self.DCA([x2_t, x3_t, x4_t])
        cfm_feature1, cfm_feature2 = self.CFM(x4_t, x3_t, x2_t, x2_d, x3_d)

        # SAM
        T2 = self.Translayer2_0(ctm_feature)
        T2 = self.down05(T2)
        sam_feature1 = self.tem(cfm_feature1, T2)  # main
        sam_feature2 = self.tem(cfm_feature2, T2)
        # sam_feature3 = self.tem(cfm_feature3, T2)

        prediction1 = self.out_SAM(sam_feature1)  # main
        prediction2 = self.out_SAM(sam_feature2)
        # prediction3 = self.out_SAM(sam_feature3)
        prediction3 = self.out_CFM(cfm_feature1)  # main
        prediction4 = self.out_CFM(cfm_feature2)
        # prediction6 = self.out_CFM(cfm_feature3)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction3_8 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')
        prediction4_8 = F.interpolate(prediction4, scale_factor=8, mode='bilinear')
        # prediction5_8 = F.interpolate(prediction5, scale_factor=8, mode='bilinear')
        # prediction6_8 = F.interpolate(prediction6, scale_factor=8, mode='bilinear')
        return prediction1_8, prediction2_8, prediction3_8, prediction4_8,x4_d

if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())