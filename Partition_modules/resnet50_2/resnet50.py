import torch
from .stage0 import Stage0
from .stage1 import Stage1

class resnet50(torch.nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input0):
        (out0, out1) = self.stage0(input0)
        out2 = self.stage1(out0, out1)
        return out2
