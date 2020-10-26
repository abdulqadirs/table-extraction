import torch.nn as nn

class SFCN(nn.Module):
    'Shared Fully Convolutional Network'
    def __init__(self, input_channels):
        super(SFCN, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=18, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=18, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=18, kernel_size=7, stride=1, padding=6, dilation=2, bias=False),
            nn.ReLU(True))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x