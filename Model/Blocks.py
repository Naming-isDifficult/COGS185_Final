from Model.Layers import ConvLayer, SeparatedConv, DuplicatedSeparatedConv
import torch
import torch.nn as nn

class NormalBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, batch_norm=True):
        super(NormalBlock, self).__init__()

        #initialize conv layers
        self.conv1 = ConvLayer(input_ch, output_ch, kernel_size, batch_norm)
        self.conv2 = ConvLayer(output_ch, output_ch, kernel_size, batch_norm)

        #initialize 1x1 conv if needed
        self.conv1x1 = None
        if input_ch != output_ch:
            self.conv1x1 = nn.Conv2d(input_ch, output_ch, 1, bias=False)

        #final activation
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        residual = x

        out = self.conv1((out,True))
        out = self.conv2((out,False))

        if self.conv1x1:
            residual = self.conv1x1(residual)

        return self.relu(out+residual)
    
class SeparatedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, batch_norm=True):
        super(SeparatedBlock, self).__init__()

        #initialize conv layers
        self.conv1 = SeparatedConv(input_ch, output_ch, kernel_size, batch_norm)
        self.conv2 = SeparatedConv(output_ch, output_ch, kernel_size, batch_norm)

        #initialize 1x1 conv if needed
        self.conv1x1 = None
        if input_ch != output_ch:
            self.conv1x1 = nn.Conv2d(input_ch, output_ch, 1, bias=False)

        #final activation
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        residual = x

        out = self.conv1((out,True))
        out = self.conv2((out,False))

        if self.conv1x1:
            residual = self.conv1x1(residual)

        return self.relu(out+residual)

class DuplicatedSeparatedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, batch_norm):
        super(DuplicatedSeparatedBlock, self).__init__()

        #initialize conv layers
        self.conv1 = DuplicatedSeparatedConv(input_ch, output_ch,\
                                             kernel_size, batch_norm)
        self.conv2 = DuplicatedSeparatedConv(output_ch, output_ch,\
                                             kernel_size, batch_norm)

        #initialize 1x1 conv if needed
        self.conv1x1 = None
        if input_ch != output_ch:
            self.conv1x1 = nn.Conv2d(input_ch, output_ch, 1, bias=False)

        #final activation
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        residual = x

        out = self.conv1((out,True))
        out = self.conv2((out,False))

        if self.conv1x1:
            residual = self.conv1x1(residual)

        return self.relu(out+residual)