import torch
import torch.nn as nn
from Model.Layers import ConvLayer
from Model.Blocks import NormalBlock, SeparatedBlock, DuplicatedSeparatedBlock

def get_resnet18(type, input_ch, input_size, output_dim, batch_norm=True):
    block = None
    if type == 'normal':
        block = NormalBlock
    elif type == 'separated':
        block = SeparatedBlock
    elif type == 'duplicated':
        block = DuplicatedSeparatedBlock
    else:
        raise ValueError('Unsupported type')

    return ResNet18(block, input_ch, input_size, output_dim, batch_norm)

class ResNet18(nn.Module):
    def __init__(self, Block, input_ch, input_size, output_dim, batch_norm=True):
        super(ResNet18, self).__init__()

        #conv 7x7, 64, /2
        #out = input_size // 2
        self.conv1 = ConvLayer(input_ch, 64, 7, 2, batch_norm)

        #max pool, 2x2, stride 2
        #out = input_size // 4
        self.max_pool = nn.MaxPool2d(2, stride=2)
        
        #conv block 2 (2 blocks, 3x3, 64, stride = [1, 1])
        #out = input_size
        self.conv2_1 = Block(64, 64, 3, 1, batch_norm)
        self.conv2_2 = Block(64, 64, 3, 1, batch_norm)

        #conv block 3 (2 blocks, 3x3, 128, stride = [2, 1])
        #out = input_size //8
        self.conv3_1 = Block(64, 128, 3, 2, batch_norm)
        self.conv3_2 = Block(128, 128, 3, 1, batch_norm)

        #conv block 4 (2 blocks, 3x3, 256, stride = [2, 1])
        #out = input_size //16
        self.conv4_1 = Block(128, 256, 3, 2, batch_norm)
        self.conv4_2 = Block(256, 256, 3, 1, batch_norm)

        #conv block 5 (2 blocks, 3x3, 256, stride = [2, 1])
        #out = input_size //32
        self.conv5_1 = Block(256, 512, 3, 2, batch_norm)
        self.conv5_2 = Block(512, 512, 3, 1, batch_norm)

        #average pool
        #out = [batch, 512, 1, 1]
        self.average_pool = nn.AdaptiveAvgPool2d((1,1))

        #dense layer
        #out = output_dim
        self.fc = nn.Linear(in_features=512, out_features=output_dim)

    def forward(self, x):
        out = x

        #conv block 1
        out = self.conv1((out,True))
        out = self.max_pool(out)

        #conv block 2
        out = self.conv2_1(out)
        out = self.conv2_2(out)

        #conv block 3
        out = self.conv3_1(out)
        out = self.conv3_2(out)

        #conv block 4
        out = self.conv4_1(out)
        out = self.conv4_2(out)

        #conv block 5
        out = self.conv5_1(out)
        out = self.conv5_2(out)

        #classification
        out = self.average_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out