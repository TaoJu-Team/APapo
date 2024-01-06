import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4
from .stage5 import Stage5
from .stage6 import Stage6
from .stage7 import Stage7

class resnet50(torch.nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()
        self.stage5 = Stage5()
        self.stage6 = Stage6()
        self.stage7 = Stage7()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input0):
        (out1, out0) = self.stage0(input0)
        (out3, out2) = self.stage1(out1, out0)
        (out4, out5) = self.stage2(out3, out2)
        (out6, out7) = self.stage3(out4, out5)
        (out9, out8) = self.stage4(out6, out7)
        (out11, out10) = self.stage5(out9, out8)
        (out12, out13) = self.stage6(out11, out10)
        out14 = self.stage7(out12, out13)
        return out14
