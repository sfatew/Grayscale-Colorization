
import torch.nn as nn

from model_build.Base import BaseColor
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class ColorizationNet(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ColorizationNet, self).__init__()

        vgg = vgg16_bn(weights= VGG16_BN_Weights.IMAGENET1K_V1)
        vgg_features = vgg.features

        # Replace model1 to model4 with the first 16 layers from VGG16
        # These correspond to conv1_1 to conv4_3
        self.encoder = nn.Sequential(*list(vgg_features.children())[:33])  

        
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]


        self.decoder = nn.Sequential(
            *(model5 + model6 + model7 + model8)
        )
        #self.softmax = nn.Softmax(dim=1)
        #self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        #self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        encoder = self.encoder(self.imageNet_normalize(self.normalize_l(input_l)))
        decoder = self.decoder(encoder)
        #out_reg = self.model_out(self.softmax(conv8_3))

        #return self.unnormalize_ab(self.upsample4(out_reg))
        #return self.softmax(conv8_3)
        return decoder
    
def colorization():
    model = ColorizationNet()
    return model

if __name__ == "__main__":
    model = vgg16_bn(weights= VGG16_BN_Weights.IMAGENET1K_V1)
    for name, layer in model.named_modules():
        print(name, layer)