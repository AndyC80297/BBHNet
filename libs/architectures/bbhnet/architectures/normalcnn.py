import torch
import torch.nn.functional as F

from torch import nn


class ElenaCNN(nn.Module):
    
    def __init__(
        self, 
        num_ifos: int, 
        dur: int=1, 
        pool_size: int=2 
    ):
        
        super(ElenaCNN, self).__init__()
        
        nodes = [
            [num_ifos, 120], 
            [120, 80], 
            [80, 40], 
            [40, 40]
        ]

        self.pool_size = pool_size
        self.Conv_layers = nn.ModuleList()

        for node in nodes:
            # print(node[0])
            self.Conv = nn.Conv1d(
                in_channels=node[0], 
                out_channels=node[1], 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        
            self.Conv_layers.append(self.Conv)
        
        # len(self.Conv_layers)
        last_axis_size = dur*4096/self.pool_size**(len(self.Conv_layers))

        print(nodes[-1][-1]*last_axis_size)
        self.L1 = nn.Linear(int(nodes[-1][-1]*last_axis_size), 200)
        self.L2 = nn.Linear(200, 100)
        self.L3 = nn.Linear(100, 1)

    def forward(self, x):
        
        for layer in self.Conv_layers:
            
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p = 0.8)
            x = nn.MaxPool1d(kernel_size=self.pool_size)(x)

        x = torch.flatten(x, 1)

        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        # x = self.L3(x)
        # x = F.softmax(x, dim = 1)
        x = self.L3(x)

        
        return x



class UwCNN(nn.Module):
    # used three convolutional layers along with three max pool layers. The dimensions are preserved through each 
    # convolutional layer but reduced through the maxpool layers. The number of channels are doubled through each maxpool 
    # layer starting with 1 channel to 16.
    def __init__(
            self, 
            num_ifos: int,
            in_channels: int=2
        ):
        
        super(UwCNN, self).__init__()
        
        
        # First convolution layer (1 channel -> 16 channels, preserve original dimension by adding padding = 2)
        self.cnn1 = nn.Conv1d(
            in_channels=in_channels, out_channels=16, 
            kernel_size=20, stride=1, padding=2)
        
        # First max pooling layer with kernel size = 2
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
         
        # Second convolution layer (16 channel -> 32 channels, preserve dimension by adding padding = 2)
        self.cnn2 = nn.Conv1d(
            in_channels=16, out_channels=32, 
            kernel_size=20, stride=1, padding=2)
        
        # Second max pooling layer with kernel size = 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        # Second convolution layer (32 channel -> 64 channels, preserve dimension by adding padding = 2)
        self.cnn3 = nn.Conv1d(
            in_channels=32, out_channels=64, 
            kernel_size=25, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        
        self.dropout = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(31744, 1000)
        self.fc2 = nn.Linear(1000, 1)
        
    # feed forward function takes output from each layer and feeds it into next layer    
    def forward(self, x):
        
        # input image -> conv1 -> relu -> maxpool1
        conv1_out = F.relu(self.cnn1(x))       
        pool1_out = self.maxpool1(conv1_out)
        
        # maxpool1 output -> conv2 -> relu -> maxpool2
        conv2_out = F.relu(self.cnn2(pool1_out))    
        pool2_out = self.maxpool2(conv2_out)
        
        # maxpool2 output -> conv3 -> relu -> maxpool3
        conv3_out = F.relu(self.cnn3(pool2_out))
        pool3_out = self.maxpool3(conv3_out)
        
        # flatten the maxpool2 output to be used as input into FCN layer
        fcn_input = pool3_out.view(pool3_out.size(0), -1)
    
        # Use the raw output of the fully connected layer as the final output
        output = self.fc1(fcn_input)
        output = self.dropout(output)
        output = self.fc2(output)

        
        return output