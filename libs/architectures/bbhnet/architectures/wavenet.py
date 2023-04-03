import torch

import torch.nn.functional as F

from torch import nn


class WaveNet(nn.Module):
    
    def __init__(
        self, 
        num_ifos: int,
        c_depth: int=8, 
        n_chann: int=64, 
        l1: int=1024, 
        l2: int=128
    ):
        
        super(WaveNet, self).__init__()
        
        self.c_depth = c_depth
        self.n_chann = n_chann
        
        self.Conv_In = nn.Sequential(
            # nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=num_ifos, 
                      out_channels=self.n_chann, 
                      kernel_size=2)
        )
        
        self.Conv_Out = nn.Conv1d(in_channels=self.n_chann, 
                                  out_channels=1, 
                                  kernel_size=1)

        self.WaveNet_layers = nn.ModuleList()
        
        
        for i in range(self.c_depth):

            conv_layer = nn.Conv1d(
                in_channels=self.n_chann, 
                out_channels=self.n_chann,
                kernel_size=2,
                dilation=2**i
            )
            
            self.WaveNet_layers.append(conv_layer)
        
        
        # self.L1 = nn.Linear(8192-2**c_depth, l1)
        self.L1 = nn.Linear(3840, l1)
        self.L2 = nn.Linear(l1, l2)
        self.L3 = nn.Linear(l2, 1)

        
    def forward(self, x):

        x = self.Conv_In(x)
        x = F.relu(x)
        
        for what_are_u_wavin_at in self.WaveNet_layers:
            x = what_are_u_wavin_at(x)
            x = F.relu(x)
            
        x = self.Conv_Out(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        # x = F.softmax(self.L3(x), dim = 1)
        x = self.L3(x)
        
        return x