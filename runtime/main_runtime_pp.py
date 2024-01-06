# -*- coding:utf-8 -*-
# Date:2022/6/17
# Description:
import json

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
import sys

local_rank = 0
module = "DP_Partition.Partition_modules.resnet18"
batch_size = 16
eval_batch_size = 16
config_path = "../Partition_modules/resnet18/mp_conf.json"
distributed_backend = "gloo"
fp16 = True
loss_scale = 1
master_addr = 'localhost'
rank = 0
num_ranks_in_server = 1  # 每台机器的gpu数量
checkpoint_dir_not_nfs = True
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
forward_only = False
checkpoint_dir = ''


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


# Helper methods.
def is_first_stage():
    return stage is None or (stage == 0)


def is_last_stage():
    return stage is None or (stage == (num_stages - 1))


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


def main():
    global best_prec1, module, synthetic_data

    # torch.cuda.set_device(local_rank)
    device = ['cuda:' + str(i) for i in range(torch.cuda.device_count())]
    print(device)

    # 定义损失函数(criterion)
    criterion = nn.CrossEntropyLoss()

    # 创建模型的阶段, 加载,生成模型
    module = importlib.import_module(module)
    arch = module.arch()
    model = module.model(criterion)  # 依据模块来构建模型

    # 确定传入模型中所有张量的形状
    input_size = [batch_size, 3, 224, 224]
    training_tensor_shapes = {"input0": input_size, "target": [batch_size]}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input": 0}
    target_tensor_names = {"target"}
    # 遍历模型的每个层（跳过最后loss层）
    for cuda_id, (stage, inputs, outputs) in enumerate(model[:-1]):  # Skip last layer (loss).
        print(cuda_id)
        input_tensors = []
        # 遍历每层的输入，构建输入张量
        for input in inputs:
            input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
                                       dtype=torch.float32).to(device[cuda_id])
            input_tensors.append(input_tensor)
        stage.to(device[cuda_id])
        # PyTorch 不应为合成输入的反向传递维护元数据。如果没有以下行，则在完整 DP 配置中运行时间会慢 1.5 倍。
        with torch.no_grad():  # 所有计算得出的tensor的requires_grad都自动设置为False,即不会自动求导
            # 通过调用stage对应的forward函数，构建出输出
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        # 遍历每层的输出，设置其类型和形状
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    # 构建输出值张量类型
    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            [eval_batch_size] + training_tensor_shapes[key][1:])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if config_path is not None:
        json_config_file = json.load(open(config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    r = runtime.StageRuntime(
        model=model, distributed_backend=distributed_backend,
        fp16=fp16, loss_scale=loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=master_addr, rank=rank,
        local_rank=local_rank,
        num_ranks_in_server=num_ranks_in_server,
        verbose_freq=verbose_frequency,
        model_type=runtime.IMAGE_CLASSIFICATION,
        enable_recompute=recompute)

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    stage = r.stage
    num_stages = r.num_stages
    num_ranks = r.num_ranks
    if not is_first_stage():
        synthetic_data = True

    # define optimizer
    if no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    # if args.resume:
    #     checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
    #     assert os.path.isfile(checkpoint_file_path)
    #     print("=> loading checkpoint '{}'".format(checkpoint_file_path))
    #     checkpoint = torch.load(checkpoint_file_path)
    #     args.start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     r.load_state_dict(checkpoint['state_dict'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(checkpoint_file_path, checkpoint['epoch']))

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
                                          r.model_parameters, loss_scale,
                                          num_versions=num_versions,
                                          lr=lr,
                                          momentum=momentum,
                                          weight_decay=weight_decay,
                                          verbose_freq=verbose_frequency,
                                          macrobatch=False)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True  # 实现网络的加速

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if arch == 'inception_v3':
        if synthetic_data:
            train_dataset = SyntheticDataset((3, 299, 299), 10000)
        else:
            traindir = os.path.join(data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
    else:
        if synthetic_data:
            train_dataset = SyntheticDataset((3, 224, 224), 1000000)
        else:
            traindir = os.path.join(data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

    if synthetic_data:
        val_dataset = SyntheticDataset((3, 224, 224), 10000)
    else:
        valdir = os.path.join(data_dir, 'val')
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=rank)
            distributed_sampler = True
    #
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    # if checkpoint is loaded, start by running validation
    if resume:
        assert start_epoch > 0
        validate(val_loader, r, start_epoch - 1)

    for epoch in range(start_epoch, epochs):
            if distributed_sampler:
                train_sampler.set_epoch(epoch)

            # train or run forward pass only for one epoch
            if forward_only:
                validate(val_loader, r, epoch)
            else:
                train(train_loader, r, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, r, epoch)
            if r.stage != r.num_stages: prec1 = 0

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            should_save_checkpoint = checkpoint_dir_not_nfs or r.rank_in_stage == 0
            if checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': r.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_dir, r.stage)


if __name__ == '__main__':
    main()
