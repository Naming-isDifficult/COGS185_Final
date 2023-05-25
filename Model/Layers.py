import torch
import torch.nn as nn

'''
  A typical convolutional layer
'''
class ConvLayer(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, batch_norm=True):
        super(ConvLayer, self).__init__()

        #remove bais term if batch_norm is applied
        use_bias = not batch_norm

        #initialize all layers
        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size,\
                              padding='same', bias=use_bias)
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
          x will be a 2-tuple
          The first element will be the input
          The second element indicates using activation
          at the end or not.
        '''
        out = x[0]
        final_activation = x[1]

        out = self.conv(out)
        if self.bn:
            out = self.bn(out)
        out = self.relu(out) if final_activation\
                    else out

        return out

'''
  A spatial separated convolutional layer.
  It can be either height first or width first depending on the given
  parameter
  E.g.
    If the kernel is [3,3] and it's height first, the kernel will be
    seperated into [3,1] followed by [1,3].
    If the kernel is [3,3] and it's width first, the kernel will be
    seperated into [1,3] followed by [3,1].
'''
class SeparatedConv(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size,\
                 batch_norm=True, height_first=True):
        super(SeparatedConv, self).__init__()

        #remove bais term if batch_norm is applied
        use_bias = not batch_norm

        #find kernel height and width
        try:
            kernel_size = list(kernel_size)
        except:
            kernel_size = [kernel_size]
        kernel_height = kernel_size[0]
        kernel_width = kernel_size[-1]

        #obtain kernel size
        first_kernel = (kernel_height, 1) if height_first\
                            else (1, kernel_width)
        second_kernel = (1, kernel_width) if height_first\
                            else (kernel_height, 1)

        #initialize first conv layers
        self.conv1 = nn.Conv2d(input_ch, output_ch, first_kernel,\
                              padding='same', bias=use_bias)
        self.bn1 = None
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(output_ch)
        self.relu1 = nn.ReLU()

        #initialze second conv layers
        self.conv2 = nn.Conv2d(output_ch, output_ch, second_kernel,\
                               padding='same', bias=use_bias)
        self.bn2 = None
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(output_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        '''
          x will be a 2-tuple
          The first element will be the input
          The second element indicates using activation
          at the end or not.
        '''
        out = x[0]
        final_activation = x[1]
        
        #go through first conv layer
        out = self.conv1(out)
        if self.bn1:
            out = self.bn1(out)
        out = self.relu1(out)

        #go throug second conv layer
        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)
        out = self.relu2(out) if final_activation\
                    else out

        return out

'''
  A spatial sperated convolution layer with two paths.
  Each path is a spatial seperated convolution. One path should be
  height-first and the other should be width first.
  The output of this layer should be the sum of the output from both path.
'''
class DuplicatedSeparatedConv(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, batch_norm=True):
        super(DuplicatedSeparatedConv, self).__init__()

        #initialize both paths
        self.left_path = SeparatedConv(input_ch, output_ch,\
                                       kernel_size,batch_norm=batch_norm,\
                                       height_first=True)
        self.right_path = SeparatedConv(input_ch, output_ch,\
                                       kernel_size,batch_norm=batch_norm,\
                                       height_first=False)

    def forward(self, x):
        out_left = self.left_path(x)
        out_right = self.right_path(x)

        return out_left+out_right

        