from .resnet18 import resnet18
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "resnet18"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out1", "out0"]),
        (Stage1(), ["out1", "out0"], ["out3", "out2"]),
        (Stage2(), ["out3", "out2"], ["out5", "out4"]),
        (Stage3(), ["out5", "out4"], ["out6"]),
        (criterion, ["out6"], ["loss"])
    ]

def full_model():
    return resnet18()
