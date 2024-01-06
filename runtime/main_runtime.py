# -*- coding:utf-8 -*-
# Date:2022/6/17
# Description:
import torch
import torch.nn as nn
import importlib
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
import time
import shutil
from apex import amp

local_rank = 0
module = "DP_Partition.Partition_modules.resnet18"
batch_size = 16
eval_batch_size = 16
config_path = "../Partition_modules/"
distributed_backend = "nccl"
fp16 = True
loss_scale = 1
master_addr = ''
rank = 1
num_ranks_in_server = 1  # 每台机器的gpu数量
verbose_frequency = True
recompute = True  # 从向前传递重新计算张量，而不是保存它们
synthetic_data = True  # 使用合成数据
stage = None
num_stages = None
no_input_pipelining = True
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
data_dir = '../data/kaggle_cifar10_tiny'
start_epoch = 0
epochs = 90
resume = None
num_minibatches = None
workers = 4
print_freq = 10  # 打印频率
best_prec1 = 0


# amp_handle = amp.init(enabled=fp16)


class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    global lr
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    epoch_start_time = time.time()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if num_minibatches is not None and i >= num_minibatches:
                break
            target = target.cuda(non_blocking=True)

            input = input.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                n = len(val_loader)
                if num_minibatches is not None:
                    n = num_minibatches
                print('Test: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, n, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5,
                    memory=(float(torch.cuda.memory_allocated()) / 10 ** 9),
                    cached_memory=(float(torch.cuda.memory_cached()) / 10 ** 9)))
                import sys;
                sys.stdout.flush()

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    # criterion=nn.CrossEntropyLoss()

    # 加载，生成模型
    global best_prec1
    modules = importlib.import_module(module)
    arch = modules.arch()

    # 依据模块来构建模型
    model = modules.full_model()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    cudnn.benchmark = True

    # 数据加载代码
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if arch == 'inception_v3':
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
        )
        if synthetic_data:
            train_dataset = SyntheticDataset((3, 299, 299), len(train_dataset))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if synthetic_data:
            train_dataset = SyntheticDataset((3, 224, 224), len(train_dataset))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=eval_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_dict = {
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_dict, is_best)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    epoch_start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        if num_minibatches is not None and i >= num_minibatches:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        input = input.cuda()
        # compute output
        output = model(input)
        if isinstance(output, tuple):
            loss = sum((criterion(output_elem, target) for output_elem in output))
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0], target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            n = len(train_loader)
            if num_minibatches is not None:
                n = num_minibatches
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, n, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                memory=(float(torch.cuda.memory_allocated()) / 10 ** 9),
                cached_memory=(float(torch.cuda.memory_cached()) / 10 ** 9)))
            import sys;
            sys.stdout.flush()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


if __name__ == '__main__':
    main()
