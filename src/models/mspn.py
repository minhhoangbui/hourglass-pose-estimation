import torch.nn as nn
import torch.nn.functional as F
from src.loss.mse import MSELoss

__all__ = ['mspn']


class ConvBNReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
                 has_relu=True, mobile=False):
        super(ConvBNReLu, self).__init__()
        if mobile:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=out_planes)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        self.has_relu = has_relu
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, mobile=False):
        super(Bottleneck, self).__init__()
        self.conv_bn_relu1 = ConvBNReLu(in_planes, planes, kernel_size=1,
                                        stride=1, padding=0, has_relu=True)
        self.conv_bn_relu2 = ConvBNReLu(planes, planes, kernel_size=3,
                                        stride=stride, padding=1, has_relu=True,
                                        mobile=mobile)
        self.conv_bn_relu3 = ConvBNReLu(planes, planes * self.expansion,
                                        kernel_size=1, stride=1, padding=0,
                                        has_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)
        return out


class InputLayer(nn.Module):
    def __init__(self):
        super(InputLayer, self).__init__()
        self.conv = ConvBNReLu(3, 64, kernel_size=7, stride=2, padding=3,
                               has_relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class DownSample(nn.Module):
    def __init__(self, block, layers, has_skip=False,
                 zero_init_residual=False, mobile=False):
        super(DownSample, self).__init__()
        self.has_skip = has_skip
        self.mobile = mobile
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = ConvBNReLu(self.in_planes, planes * block.expansion,
                                    kernel_size=1, stride=stride, padding=0,
                                    has_relu=False)

        layers = list()
        layers.append(block(self.in_planes, planes, stride, downsample, self.mobile))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, skip1, skip2):
        x1 = self.layer1(x)
        if self.has_skip:
            x1 = x1 + skip1[0] + skip2[0]
        x2 = self.layer2(x1)
        if self.has_skip:
            x2 = x2 + skip1[1] + skip2[1]
        x3 = self.layer3(x2)
        if self.has_skip:
            x3 = x3 + skip1[2] + skip2[2]
        x4 = self.layer4(x3)
        if self.has_skip:
            x4 = x4 + skip1[3] + skip2[3]
        return x4, x3, x2, x1


class UpsampleUnit(nn.Module):

    def __init__(self, ind, in_planes, up_size, output_chl_num, output_shape,
                 chl_num=256, gen_skip=False, gen_cross_conv=False, mobile=False):
        super(UpsampleUnit, self).__init__()
        self.output_shape = output_shape
        self.u_skip = ConvBNReLu(in_planes, chl_num, kernel_size=1, stride=1,
                                 padding=0, has_relu=False)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_size = up_size
            self.up_conv = ConvBNReLu(chl_num, chl_num, kernel_size=1,
                                      stride=1, padding=0, has_relu=False,
                                      mobile=mobile)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.skip1 = ConvBNReLu(in_planes, in_planes, kernel_size=1,
                                    stride=1, padding=0, has_relu=True,
                                    mobile=mobile)
            self.skip2 = ConvBNReLu(chl_num, in_planes, kernel_size=1,
                                    stride=1, padding=0, has_relu=True)

        self.gen_cross_conv = gen_cross_conv
        if self.ind == 3 and self.gen_cross_conv:
            self.cross_conv = ConvBNReLu(chl_num, 64, kernel_size=1,
                                         stride=1, padding=0, has_relu=True)

        self.res_conv1 = ConvBNReLu(chl_num, chl_num, kernel_size=1,
                                    stride=1, padding=0, has_relu=True,
                                    mobile=mobile)

        self.res_conv2 = ConvBNReLu(chl_num, output_chl_num, kernel_size=3,
                                    stride=1, padding=1, has_relu=False)

    def forward(self, x, up_x=None):
        out = self.u_skip(x)

        if self.ind > 0:
            up_x = F.interpolate(up_x, size=self.up_size, mode='bilinear',
                                 align_corners=True)
            up_x = self.up_conv(up_x)
            out += up_x
        out = self.relu(out)

        res = self.res_conv1(out)
        res = self.res_conv2(res)
        res = F.interpolate(res, size=self.output_shape, mode='bilinear',
                            align_corners=True)

        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.skip1(x)
            skip2 = self.skip2(out)

        cross_conv = None
        if self.ind == 3 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, res, skip1, skip2, cross_conv


class Upsample(nn.Module):
    def __init__(self, output_chl_num, output_shape, chl_num=256,
                 gen_skip=False, gen_cross_conv=False, mobile=False):
        super(Upsample, self).__init__()
        in_planes = [2048, 1024, 512, 256]
        h = w = output_shape
        up_sizes = [(h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w)]

        self.up1 = UpsampleUnit(0, in_planes[0], up_sizes[0],
                                output_chl_num=output_chl_num, output_shape=output_shape,
                                chl_num=chl_num, gen_skip=gen_skip,
                                gen_cross_conv=gen_cross_conv, mobile=mobile)
        self.up2 = UpsampleUnit(1, in_planes[1], up_sizes[1],
                                output_chl_num=output_chl_num, output_shape=output_shape,
                                chl_num=chl_num, gen_skip=gen_skip,
                                gen_cross_conv=gen_cross_conv, mobile=mobile)
        self.up3 = UpsampleUnit(2, in_planes[2], up_sizes[2],
                                output_chl_num=output_chl_num, output_shape=output_shape,
                                chl_num=chl_num, gen_skip=gen_skip,
                                gen_cross_conv=gen_cross_conv, mobile=mobile)
        self.up4 = UpsampleUnit(3, in_planes[3], up_sizes[3],
                                output_chl_num=output_chl_num, output_shape=output_shape,
                                chl_num=chl_num, gen_skip=gen_skip,
                                gen_cross_conv=gen_cross_conv, mobile=mobile)

    def forward(self, x4, x3, x2, x1):
        out1, res1, skip1_1, skip2_1, _ = self.up1(x4)
        out2, res2, skip1_2, skip2_2, _ = self.up2(x3, out1)
        out3, res3, skip1_3, skip2_3, _ = self.up3(x2, out2)
        out4, res4, skip1_4, skip2_4, cross_conv = self.up4(x1, out3)

        # 'res' starts from small size
        res = [res1, res2, res3, res4]
        skip1 = [skip1_4, skip1_3, skip1_2, skip1_1]
        skip2 = [skip2_4, skip2_3, skip2_2, skip2_1]

        return res, skip1, skip2, cross_conv


class SingleStage(nn.Module):
    def __init__(self, output_chl_num, output_shape, has_skip=False,
                 gen_skip=False, gen_cross_conv=False, chl_num=256,
                 zero_init_residual=False, mobile=False):
        super(SingleStage, self).__init__()
        self.zero_init_residual = zero_init_residual
        self.layers = [3, 4, 6, 3]
        self.downsample = DownSample(Bottleneck, self.layers,
                                     has_skip, zero_init_residual,
                                     mobile=mobile)

        self.upsample = Upsample(output_chl_num, output_shape,
                                 chl_num, gen_skip,
                                 gen_cross_conv,
                                 mobile=mobile)
        print('down_sample', sum(p.numel() for p in self.downsample.parameters() if p.requires_grad))
        print('up_sample', sum(p.numel() for p in self.upsample.parameters() if p.requires_grad))

    def forward(self, x, skip1, skip2):
        x4, x3, x2, x1 = self.downsample(x, skip1, skip2)
        res, skip1, skip2, cross_conv = self.upsample(x4, x3, x2, x1)
        return res, skip1, skip2, cross_conv


class MSPN(nn.Module):
    def __init__(self, num_stacks, num_classes, out_res, up_channel_num, mobile=False):
        super(MSPN, self).__init__()
        self.top = InputLayer()
        self.stage_num = num_stacks
        self.output_chl_num = num_classes
        self.output_shape = out_res
        self.upsample_chl_num = up_channel_num
        mspn_modules = list()
        for i in range(self.stage_num):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.stage_num - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False
                gen_cross_conv = False
            stage = SingleStage(
                        self.output_chl_num, self.output_shape,
                        has_skip=has_skip, gen_skip=gen_skip,
                        gen_cross_conv=gen_cross_conv,
                        chl_num=self.upsample_chl_num,
                        mobile=mobile
                    )
            mspn_modules.append(stage)
            print('single', sum(p.numel() for p in stage.parameters() if p.requires_grad))
        self.mspn_modules = nn.ModuleList(mspn_modules)

        self.criterion = MSELoss(use_target_weight=True)

    def forward(self, x):
        x = self.top(x)
        skip1 = None
        skip2 = None
        outputs = list()
        for i in range(self.stage_num):
            res, skip1, skip2, x = self.mspn_modules[i](x, skip1, skip2)
            outputs.append(res)
        return outputs

    def compute_loss(self, outputs, target, target_weight):
        loss = 0
        for outs in outputs:
            for o in outs:
                loss += self.criterion(o, target, target_weight)
        return loss


def mspn(**kwargs):
    model = MSPN(num_stacks=kwargs['num_stacks'], num_classes=kwargs['num_classes'],
                 out_res=kwargs['out_res'], mobile=kwargs['mobile'],
                 up_channel_num=kwargs['num_blocks'])
    return model


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    dummy_variable = torch.ones((1, 3, 256, 256))
    model = MSPN(num_stacks=2, num_classes=16, out_res=64, mobile=True, up_channel_num=256)
    summary(model, (3, 256, 256), device='cpu')
    model(dummy_variable)

