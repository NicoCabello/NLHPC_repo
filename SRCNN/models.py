from torch import nn

class ConvolutionalBlock(nn.Module):
    """
    A convolutional block ti construct the network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=None, dropout=None):
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'selu', 'leakyrelu', 'tanh'}

        # layers holder
        layers = list()

        # convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        )

        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'selu':
            layers.append(nn.SELU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLu(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input)     # (N, out_channels, w, h)
        return output


class SRCNN(nn.Module):

    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = ConvolutionalBlock(in_channels=num_channels, out_channels=576, kernel_size=3, activation='SELU', dropout=0.2)
        self.conv2 = ConvolutionalBlock(in_channels=576, out_channels=192, kernel_size=3, activation='SELU', dropout=0.2)
        self.conv3 = ConvolutionalBlock(in_channels=192, out_channels=64, kernel_size=3, activation='SELU', dropout=0.2)
        self.conv4 = ConvolutionalBlock(in_channels=64, out_channels=1, kernel_size=3)
        self.fc = nn.Linear(128, 128)

    def forward(self, lr_imgs):
        output = self.conv1(lr_imgs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.fc(output)
        return output

# class SRCNN(nn.Module):
#     def __init__(self, num_channels=1):
#         super(SRCNN, self).__init__()
#         self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
#         self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
#         return x