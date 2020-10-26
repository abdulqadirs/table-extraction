import torch
import numpy as np
import torch.nn as nn
from config import Config

class ProjectionPooling(nn.Module):
  def __init__(self, direction):
    super(ProjectionPooling, self).__init__()
    self.direction = direction
  
  def forward(self, x):
    #batch, channels, height, width = x.size()
    device = Config.get('device')
    input_mask = torch.from_numpy(np.ones(x.size())).type(torch.FloatTensor).to(device)
    if self.direction == 1:
      output = torch.mean(x, 2).unsqueeze(2) * input_mask 
      del input_mask
      torch.cuda.empty_cache()
      return output
    elif self.direction == 0:
      output = torch.mean(x, 3).unsqueeze(3) * input_mask
      del input_mask
      torch.cuda.empty_cache()
      return output