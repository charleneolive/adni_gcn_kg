import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


'''
https://github.com/okankop/Efficient-3DCNNs
3D convolutional network: https://drive.google.com/drive/folders/1eggpkmy_zjb62Xra6kQviLa67vzP_FR8
'''

    
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)
        
        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
        	out += x
        out = self.relu(out)

        return out
    
class SqueezeNet(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,			
    	         version=1.1,
    	         num_classes=600):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv3d(3, 96, kernel_size=7, stride=(1,2,2), padding=(3,3,3)),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192, use_bypass=True),
                Fire(384, 64, 256, 256),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(512, 64, 256, 256, use_bypass=True),
            )
        if version == 1.1:
            self.features = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, stride=(1,2,2), padding=(1,1,1)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256, use_bypass=True),
            )
        # Final convolution is initialized differently form the rest
#         final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             final_conv,
#             nn.ReLU(inplace=True),
#             nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
#         )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
#         x = self.classifier(x)
        return x.view(x.size(0), -1)


class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward,self).__init__()
        self.model = nn.Linear(1,32)
        
    def forward(self, input1):
        output1 = self.model(input1)
        
        return output1
    
class SqueezeNetwork(nn.Module):
    def __init__(self, sample_size, sample_duration):
        super(SqueezeNetwork,self).__init__()
        s_checkpoint = torch.load("./pretrained_model/kinetics_squeezenet_RGB_16_best.pth")
        s_dict = OrderedDict()
        for k, v in s_checkpoint['state_dict'].items():
            if "classifier" not in k:
                name = k[7:] # remove `module.`
                s_dict[name] = v
                
        self.s_model = SqueezeNet(sample_size=sample_size, sample_duration=sample_duration)
        self.s_model.load_state_dict(s_dict)
        
    def forward(self, x):
        output = self.s_model(x)
        
        return output
    
class SqueezeNetwork2(nn.Module):
    '''
    with fully connected layer to reduce dimensionality
    '''
    def __init__(self, sample_size, sample_duration, emb):
        super(SqueezeNetwork2,self).__init__()
        s_checkpoint = torch.load("./pretrained_model/kinetics_squeezenet_RGB_16_best.pth")
        s_dict = OrderedDict()
        self.emb = emb
        for k, v in s_checkpoint['state_dict'].items():
            if "classifier" not in k:
                name = k[7:] # remove `module.`
                s_dict[name] = v
                
        self.s_model = SqueezeNet(sample_size=sample_size, sample_duration=sample_duration)
        self.s_model.load_state_dict(s_dict)
        self.linear = nn.Linear(65536, emb)
        
    def forward(self, x):
        output = self.s_model(x)
        output = self.linear(output.view(output.size(0), -1))
        
        return output

