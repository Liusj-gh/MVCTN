import torch
from torch import nn
from torch.nn import functional as F
from multi_view import ResNet
from multi_view import se_resnet

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, fusion='addition'):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.fusion = fusion

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d


        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=False),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=False)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        if fusion == 'concatenation':
            self.fs_conv = conv_nd(in_channels=self.in_channels*2, out_channels=self.in_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
            self.fs_bn = bn(self.in_channels)
            self.fs_relu = nn.ReLU(inplace=True)

        ### weights initialization ###
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if not bn_layer:
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        ### weights initialization ###

    def forward(self, x, y, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param y: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)
        g_y = self.g(y).view(batch_size, self.inter_channels, -1)
        g_y = g_y.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_y)
        f_div_C = F.softmax(f, dim=-1)
        fxy = torch.matmul(f_div_C, g_y)
        fxy = fxy.permute(0, 2, 1).contiguous()
        fxy = fxy.view(batch_size, self.inter_channels, *x.size()[2:])
        W_xy = self.W(fxy)
        if self.fusion == 'addition':
            z = W_xy + x
        elif self.fusion == 'concatenation':
            z = torch.cat([x, W_xy], dim=1)
            z = self.fs_conv(z)
            z = self.fs_bn(z)
            z = self.fs_relu(z)
        elif self.fusion == 'max':
            z = torch.max(W_xy, x)

        if return_nl_map:
            return z, f_div_C
        return z



class Unilateral_Fusion_Network(nn.Module):

    def __init__(self, backbone, attention, fusion='max', direction='fusion_tra_c', auxiliary=False):
        super(Unilateral_Fusion_Network, self).__init__()
        """
        :param backbone: se_resnet.se_resnet50  or  ResNet.resnet50
        """
        self.attention = attention
        self.direction = direction
        self.auxiliary = auxiliary

        self.model_P = backbone(pretrained=False, attention=attention, primary=True)
        self.model_C = backbone(pretrained=False, attention=attention, primary=True)
        self.model = backbone(pretrained=False, attention=attention, primary=False, compression=True)


        if self.direction == 'fusion_tra_p':
            self.NLB_C2P1 = NonLocalBlock(in_channels=256, sub_sample=True, fusion=fusion)
            self.NLB_C2P2 = NonLocalBlock(in_channels=512, sub_sample=True, fusion=fusion)
            self.NLB_C2P3 = NonLocalBlock(in_channels=1024, sub_sample=False, fusion=fusion)

        elif self.direction == 'fusion_tra_c':
            self.NLB_P2C1 = NonLocalBlock(in_channels=256, sub_sample=True, fusion=fusion)
            self.NLB_P2C2 = NonLocalBlock(in_channels=512, sub_sample=True, fusion=fusion)
            self.NLB_P2C3 = NonLocalBlock(in_channels=1024, sub_sample=False, fusion=fusion)
        if self.auxiliary:
            self.aux_conv = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False)
            self.aux_avgpool = nn.AdaptiveAvgPool2d(1)
            self.aux_fc_2 = nn.Linear(1024, 2)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_p, x_c, mean=[[0, 0, 0], [0, 0, 0]]):
        att_list_p = []
        att_list_c = []

        ############################################################
        x_p, att_p = self.model_P(x_p, mean[0], extract_layer1=True)
        x_c, att_c = self.model_C(x_c, mean[1], extract_layer1=True)
        att_list_p.append(att_p), att_list_c.append(att_c)
        if self.direction == 'fusion_tra_p':
            x_p = self.NLB_C2P1(x_p, x_c)
        elif self.direction == 'fusion_tra_c':
            x_c = self.NLB_P2C1(x_c, x_p)
        # x_p, x_c = self.NLB_C2P1(x_p, x_c), self.NLB_P2C1(x_c, x_p)

        x_p, att_p = self.model_P(x_p, mean[0], extract_layer2=True)
        x_c, att_c = self.model_C(x_c, mean[1], extract_layer2=True)
        att_list_p.append(att_p), att_list_c.append(att_c)
        if self.direction == 'fusion_tra_p':
            x_p = self.NLB_C2P2(x_p, x_c)
        elif self.direction == 'fusion_tra_c':
            x_c = self.NLB_P2C2(x_c, x_p)
        # x_p, x_c = self.NLB_C2P2(x_p, x_c), self.NLB_P2C2(x_c, x_p)

        x_p, att_p = self.model_P(x_p, mean[0], extract_layer3=True)
        x_c, att_c = self.model_C(x_c, mean[1], extract_layer3=True)
        att_list_p.append(att_p), att_list_c.append(att_c)
        ############################################################

        if self.direction == 'fusion_tra_p':
            x = self.NLB_C2P3(x_p, x_c)
            if self.training and self.auxiliary:
                x_c = self.aux_conv(x_c)
                x_c = self.aux_avgpool(x_c)
                x_c = x_c.reshape(x_c.size(0), -1)
                x_aux = self.aux_fc_2(x_c)
        elif self.direction == 'fusion_tra_c':
            x = self.NLB_P2C3(x_c, x_p)
            if self.training and self.auxiliary:
                x_p = self.aux_conv(x_p)
                x_p = self.aux_avgpool(x_p)
                x_p = x_p.reshape(x_p.size(0), -1)
                x_aux = self.aux_fc_2(x_p)
        x = self.model(x, extract_layer4=True)
        x = self.model(x, extract_classifier=True)

        if self.training and self.auxiliary:
            return x, x_aux, att_list_p, att_list_c
        else:
            return x, att_list_p, att_list_c

    def pretrained(self, weight_file, device):
        self._pretrained(self.model_P, weight_file['weight_file'], device)   #weight_file['weight_file_P']
        self._pretrained(self.model_C, weight_file['weight_file'], device)   #weight_file['weight_file_C']
        self._pretrained(self.model, weight_file['weight_file'], device)

    def _pretrained(self, model, weight_file, device):
        pre_dict = torch.load(weight_file, map_location=device)
        model_dict = model.state_dict()
        pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)

    def extract_features(self, x_p, x_c, **kwargs):
        if not kwargs:
            mean=[[0, 0, 0], [0, 0, 0]]
        elif 'mean' in kwargs:
            mean = kwargs['mean']

        x_p, att_p1 = self.model_P(x_p, mean[0], extract_layer1=True)
        x_c, att_c1 = self.model_C(x_c, mean[1], extract_layer1=True)
        # x_p, x_c = self.NLB_C2P1(x_p, x_c), self.NLB_P2C1(x_c, x_p)
        if self.direction == 'fusion_tra_p':
            x_p = self.NLB_C2P1(x_p, x_c)
        elif self.direction == 'fusion_tra_c':
            x_c = self.NLB_P2C1(x_c, x_p)

        x_p, att_p2 = self.model_P(x_p, mean[0], extract_layer2=True)
        x_c, att_c2 = self.model_C(x_c, mean[1], extract_layer2=True)
        # x_p, x_c = self.NLB_C2P2(x_p, x_c), self.NLB_P2C2(x_c, x_p)
        if self.direction == 'fusion_tra_p':
            x_p = self.NLB_C2P2(x_p, x_c)
        elif self.direction == 'fusion_tra_c':
            x_c = self.NLB_P2C2(x_c, x_p)

        x_p, att_p3 = self.model_P(x_p, mean[0], extract_layer3=True)
        x_c, att_c3 = self.model_C(x_c, mean[1], extract_layer3=True)

        if self.direction == 'fusion_tra_p':
            x = self.NLB_C2P3(x_p, x_c)
        elif self.direction == 'fusion_tra_c':
            x = self.NLB_P2C3(x_c, x_p)
        x = self.model(x, extract_layer4=True)
        cam = x
        x = self.model(x, extract_classifier=True)

        return x, att_p3, cam



def UF_SE_ResNet(attention, direction='fusion_tra_c', auxiliary=False):
    model = Unilateral_Fusion_Network(backbone=se_resnet.se_resnet50, attention=attention, direction=direction, auxiliary=auxiliary)
    return model

def UF_ResNet(attention, direction='fusion_tra_c'):
    model = Unilateral_Fusion_Network(backbone=ResNet.resnet50, attention=attention, direction=direction)
    return model

