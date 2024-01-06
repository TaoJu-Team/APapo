# -*- coding:utf-8 -*-
# Date:2022/6/30
# Description:
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import importlib
import time

def main():
    data_dir = '../data/kaggle_cifar10_tiny'
    module = "DP_Partition.Partition_modules.resnet18"
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 256
    workers = 4


    # criterion = nn.CrossEntropyLoss()  # 定义损失函数(criterion)
    module = importlib.import_module(module)  # 创建模型的阶段, 加载,生成模型
    model = module.full_model()
    # model = module.model(criterion)  # 依据模块来构建模型

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    cudnn.benchmark = True  # 实现网络的加速

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traindir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(0, 10):
        train(train_loader, model, optimizer)

def train(train_loader,model,optimizer):
    model.train()
    Stage0 = model.stage0.to('cuda:0')
    Stage1 = model.stage1.to('cuda:1')
    Stage2 = model.stage2.to('cuda:2')
    Stage3 = model.stage3.to('cuda:3')

    epoch_start_time=time.time()
    for i, (input, target) in enumerate(train_loader):
        Stage0(input.to('cuda:0'))


    target = target.to()

if __name__ == '__main__':
    main()
