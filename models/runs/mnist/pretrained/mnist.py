import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d_1 = torch.nn.Parameter(torch.empty([32, 1, 5, 5]))
        self.conv2d_1_bias = torch.nn.Parameter(torch.empty([32]))
        self.conv2d_2 = torch.nn.Parameter(torch.empty([64, 32, 5, 5]))
        self.conv2d_2_bias = torch.nn.Parameter(torch.empty([64]))
        self.fc_1 = torch.nn.Parameter(torch.empty([3136, 1024]))
        self.fc_1_bias = torch.nn.Parameter(torch.empty([1024]))
        self.fc_2 = torch.nn.Parameter(torch.empty([1024, 10]))
        self.fc_2_bias = torch.nn.Parameter(torch.empty([10]))

    def forward(self, inputs):
        x = F.conv2d(inputs, self.conv2d_1, padding=(2, 2))
        x = x + self.conv2d_1_bias.view(1,-1, 1, 1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), padding=0)

        x = F.conv2d(x, self.conv2d_2, padding=(2, 2))
        x = x + self.conv2d_2_bias.view(1,-1, 1, 1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), padding=0)

        x = x.permute(0, 2, 3, 1).reshape(-1, 7 * 7 * 64)
        x = torch.mm(x, self.fc_1) + self.fc_1_bias.view(1, -1)
        x = F.relu(x)
        x = torch.mm(x, self.fc_2) + self.fc_2_bias.view(1, -1)
        
        #x = F.log_softmax(x, dim=1)
        return x


class NetRescale(Net):
    def forward(self, inputs):
        inputs = (inputs * 0.5) + 0.5
        return super().forward(inputs)
    
    @property
    def net(self):
        return self

model = NetRescale()