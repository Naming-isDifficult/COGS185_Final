import torch
import torch.nn as nn


class NormalBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, batch_norm=True):
        super(NormalBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.1x1conv = None

        #remove bias term if batch_norm is applied
        use_bias = not batch_norm

        #initialize all conv layers
        self.conv_layers.extend(get_conv_layers(input_ch, output_ch,\
                                                kernel_size, batch_norm,\
                                                use_bias))

        #initialize 1x1 conv to match input_ch and output_ch
        if input_ch != output_ch:
            self.1x1conv = nn.Conv2d(input_ch, output_ch, kernel_size)

    def forward(self, x):
        out = x
        residual = x

        #compute out
        for module in self.conv_layers:
            out = module(out)

        #compute residual x
        if self.1x1conv:
            residual = self.1x1conv(residual_x)

        return out + residual

    def get_conv_layers(input_ch, output_ch, kernel_size, batch_norm, use_bias):
        '''
          A helper method to produce conv layers
        '''
        re = [nn.Conv2d(input_ch, output_ch, kernel_size,\
                        padding='same',bias=use_bias)]
        if batch_norm:
            re.append(nn.BatchNorm2d(output_ch))
        re.append(nn.ReLU())

        return re

class SeperatedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, batch_norm=True):
        super(SeperatedBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.1x1conv = None

    def get_conv_layers(input_ch, output_ch, kernel_size, batch_norm, use_bias):
        '''
          A helper method to produce conv layers
        '''
        #get the height and width of the kernel
        kernel_size = list(kernel_size)
        kernel_height = kernel_size[0]
        kernel_width = kernel[-1]

        #generate seperated kernel
        first_kernel = (kernel_height, 1)
        second_kernel = (1, kernel_width)

        #TO-DO

        return re