import numpy as np        
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

def get_optim_and_scheduler(params, args, run_cfgs):
    if not args.adam:
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9, 
                        weight_decay=run_cfgs[args.run_cfg]['weight_decay'])
    else:
        print('Adam selected, not using weight decay.')
        optimizer = torch.optim.Adam(params, args.lr, 
                        weight_decay=run_cfgs[args.run_cfg]['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                        run_cfgs[args.run_cfg]['lr_schedule'])
    return optimizer, lr_scheduler


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[1], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [Batch]: {0} [{1}/{2}]\t'
                  '| Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  '| Prec@1: {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1))
            
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[1], input.size(0))
        
        if i % 100 == 0:
            print('Test: {0}/{1}\t'
                  '| Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  '| Prec@1: {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader),  loss=losses, top1=top1))

    print('\n * Prec@1 {top1.avg:.3f}\n'.format(top1=top1))

    return top1.avg, top5.avg, losses.avg
        
        
def get_train_test_data(args):

    if args.augment:
        augmenting_transforms = [transforms.RandomCrop(32, padding=4), 
                                 transforms.RandomHorizontalFlip()]
    else:
        augmenting_transforms = []
    
    if args.dataset == "cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset == "cifar100":
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        
    tensor_and_normalize = [transforms.ToTensor(), normalize]
    transform_train = transforms.Compose([ t for t in augmenting_transforms + tensor_and_normalize])
    transform_test = transforms.Compose([*tensor_and_normalize])
        
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=transform_test)  
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.workers)

    small_loader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=args.workers,
                                             sampler=SubsetRandomSampler(range(512)))

    batch_size_1_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=args.workers,
                                             sampler=SubsetRandomSampler(range(512)))
    
    return trainloader, testloader, small_loader, batch_size_1_loader


def covariance_of_gradient_trace(batch_size_1_loader, model, criterion, optimizer):
    # switch to evaluate mode
    model.eval()
    trace = 0
    N = 512 # number of elements in loader
    def add_square_grad_to(trace):
        k = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    trace += (p.grad.detach()**2).sum().cpu().item()
                    k += p.grad.numel()
        return trace, k
    for i, (input, target) in enumerate(batch_size_1_loader):
        
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        trace, k = add_square_grad_to(trace)
    print('there were '+ str(k) + ' parameters')
    return trace/N
        
        
def pruning_scheduler(layerwise_prune_range_params, epoch):
    layerwise_prune_ranges = [range(p[0], p[1], p[2]) for p in layerwise_prune_range_params]
    iters = [r.index(epoch)+1 if epoch in r else None for r in layerwise_prune_ranges]
    n_iters = [len(r) for r in layerwise_prune_ranges]
    return iters, n_iters


def pruned_filters(model):
    '''
    counts number of pruned filters
    '''
        
    zeroed_p_count = 0
    for layer in [x for x in model.modules() if hasattr(x, 'prune_frac')]:
        zeroed_p_count += len(layer.mask) - int(layer.mask.sum())
    
    return zeroed_p_count


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
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


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res