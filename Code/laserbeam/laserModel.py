import torch
from torch import nn
import torch.nn.functional as f
import torchvision.models as models

class LaserModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = models.resnet34(pretrained=False) # v0
        self.net.conv1 = nn.Conv2d(2,64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.net.fc = nn.Linear(512, 2)



    def forward(self, x):
        return self.net(x)



#if __name__ == "__main__":
#    model = LaserModel()
#    print(model.net)