import torch.nn as nn
import torch


class EmbeddingNet(nn.Module):
    def __init__(self, in_channels):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels, 32, 5),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(1024, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 128)
                                )

    def forward(self, input1):
        output = self.convnet(input1)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class ConvolutionalNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvolutionalNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

    def forward(self, input1):
        output = self.convnet(input1)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class SiameseNetSigmoid(nn.Module):
    def __init__(self, in_channels):
        super(SiameseNetSigmoid, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels, 32, 5),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(1024 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        output1 = self.convnet(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.convnet(input2)
        output2 = output2.view(output2.size()[0], -1)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        return output
