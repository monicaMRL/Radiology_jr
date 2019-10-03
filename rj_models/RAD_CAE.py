import torch.nn as nn


class RadiologyCAE(nn.Module):
    def __init__(self):
        super(RadiologyCAE, self).__init__()

        # Convolution Layer 1
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=5)
        self.scaled_tan1 = nn.Tanh()

        # Maxpooling layer 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution Layer 2
        self.conv_layer2 = nn.Conv2d(in_channels=100, out_channels=150, kernel_size=5)
        self.scaled_tan2 = nn.Tanh()

        # Maxpooling layer 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Convolution Layer 3
        self.conv_layer3 = nn.Conv2d(in_channels=150, out_channels=200, kernel_size=3)
        self.scaled_tan3 = nn.Tanh()

        # Decoder
        # Transposed Convolution Layer 1
        self.transConv_layer1 = nn.ConvTranspose2d(in_channels=200, out_channels=150, kernel_size=3, stride=1)
        self.scaled_tan_trans1 = nn.Tanh()

        self.transConv_pooling1 = nn.ConvTranspose2d(in_channels=150, out_channels=150, kernel_size=2, stride=2)
        self.scaled_tan_trans_pooling1 = nn.Tanh()

        # Transposed Convolution Layer 2
        self.transConv_layer2 = nn.ConvTranspose2d(in_channels=150, out_channels=100, kernel_size=5, stride=1)
        self.scaled_tan_trans2 = nn.Tanh()

        self.transConv_pooling2 = nn.ConvTranspose2d(in_channels=100, out_channels=100, kernel_size=2, stride=2)
        self.scaled_tan_trans_pooling2 = nn.Tanh()

        # Transposed Convolution Layer 3
        self.transConv_layer3 = nn.ConvTranspose2d(in_channels=100, out_channels=1, kernel_size=5, stride=1)
        self.scaled_tan_trans3 = nn.Tanh()

    def forward(self, x):
        # Encode
        out = self.scaled_tan1(self.conv_layer1(x))
        # print("layer1", out.size())
        out = self.maxpool1(out)
        # print("pooling1", out.size())

        out = self.scaled_tan2(self.conv_layer2(out))
        # print("layer2", out.size())
        out = self.maxpool2(out)
        # print("pooling2", out.size())

        out = self.scaled_tan3(self.conv_layer3(out))
        # print("layer3", out.size())


        # Decode
        out = self.scaled_tan_trans1(self.transConv_layer1(out))
        out = self.scaled_tan_trans_pooling1(self.transConv_pooling1(out))

        out = self.scaled_tan_trans2(self.transConv_layer2(out))
        out = self.scaled_tan_trans_pooling2(self.transConv_pooling2(out))

        out = self.scaled_tan_trans3(self.transConv_layer3(out))

        return out
