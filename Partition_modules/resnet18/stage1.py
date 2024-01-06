import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer7 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12 = torch.nn.ReLU(inplace=True)
        self.layer13 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer14 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer17 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input0, input1):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3 + out0
        out5 = self.layer5(out3)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out10 = out10 + out5
        out12 = self.layer12(out10)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out12)
        out17 = self.layer17(out16)
        return (out15, out17)
