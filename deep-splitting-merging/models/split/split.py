import torch.nn as nn
import torch

from models.split.scfn import SFCN
from models.split.column_projection_network import CPN
from models.split.row_projection_network import RPN
from config import Config

class SplitModel(nn.Module):
    def __init__(self, input_channels):
        super(SplitModel, self).__init__()
        self.input_channels = Config.get('image_channels')

        self.sfcn = SFCN(self.input_channels)

        #self.rpn1 = RPN(input_channels=18, max_pooling=True, sigmoid=False)
        #self.rpn2 = RPN(input_channels=36, max_pooling=True, sigmoid=False)
        #self.rpn3 = RPN(input_channels=36, max_pooling=True, sigmoid=True)
        #self.rpn4 = RPN(input_channels=37, max_pooling=False, sigmoid=True)
        #self.rpn5 = RPN(input_channels=37, max_pooling=False, sigmoid=True)

        self.cpn1 = CPN(input_channels=18, max_pooling=True, sigmoid=False)
        self.cpn2 = CPN(input_channels=36, max_pooling=True, sigmoid=False)
        self.cpn3 = CPN(input_channels=36, max_pooling=True, sigmoid=True)
        self.cpn4 = CPN(input_channels=37, max_pooling=False, sigmoid=True)
        self.cpn5 = CPN(input_channels=37, max_pooling=False, sigmoid=True)

        self._init_weights()
  
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        features = self.sfcn(x)

        #r1 = self.rpn1(features)
        #r2 = self.rpn2(r1)
        #r3 = self.rpn3(r2)
        #r4 = self.rpn4(r3)
        #r5 = self.rpn5(r4)
        
        #r3 = r3[:, -1, :, :]
        #r4 = r4[:, -1, :, :]
        #r5 = r5[:, -1, :, :]

        c1 = self.cpn1(features)
        c2 = self.cpn2(c1)
        c3 = self.cpn3(c2)
        c4 = self.cpn4(c3)
        c5 = self.cpn5(c4)

        c3 = c3[:, -1, :, :]
        c4 = c4[:, -1, :, :]
        c5 = c5[:, -1, :, :]

        return [c3[:, 0, :], c4[:, 0, :], c5[:, 0, :]]
        #return (r3[:, :, 0], c3[:, 0, :])
        #return (r5[:, :, 0], c5[:, 0, :])