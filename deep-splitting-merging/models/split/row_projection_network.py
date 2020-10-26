import torch.nn as nn
import torch

from models.split.projection_pooling import ProjectionPooling

class RPN(nn.Module):
    "Row Projection Network"
    def __init__(self, input_channels, max_pooling, sigmoid):
        super(RPN, self).__init__()

        self.input_channels = input_channels
        self.max_pooling = max_pooling
        self.sigmoid = sigmoid

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=6, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.GroupNorm(3, 6), nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=6, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.GroupNorm(3, 6), nn.ReLU(True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=6, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.GroupNorm(3, 6), nn.ReLU(True))
        
        self.max_pool = nn.MaxPool2d((1, 2))

        self.conv1x1_features = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=18, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(6, 18), nn.ReLU(True))
        self.conv1x1_predictions = nn.Sequential(
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=18, out_channels=1, kernel_size=1, stride=1, bias=False))
        
        self.projection_pooling_features = ProjectionPooling(direction=0)
        self.projection_pooling_predictions = nn.Sequential(ProjectionPooling(direction=0), nn.Sigmoid())
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], 1)
        input = x
        if self.max_pooling:
            input = self.max_pool(x)

        conv1x1_features = self.conv1x1_features(input)
        projection_pooling_features = self.projection_pooling_features(conv1x1_features)
        
        if self.sigmoid:
            conv1x1_predictions = self.conv1x1_predictions(input)
            projection_pooling_predictions = self.projection_pooling_predictions(conv1x1_predictions)
            output = torch.cat([input, projection_pooling_features, projection_pooling_predictions], 1)
        else:
            output = torch.cat([input, projection_pooling_features], 1)
        
        return output