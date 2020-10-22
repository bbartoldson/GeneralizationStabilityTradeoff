import argparse
import os
import pandas as pd
import numpy as np
import socket
import torch
from hessian_eigenthings import compute_hessian_eigenthings
import models # contains various architectures and the pruning algorithms
from config import run_cfgs # contains run configurations for experiments
from utils import * # contains functions for making dataloaders, etc.

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("_") and
    not name in ['ceil', 'floor','permutation','truncnorm','resnetxx']
    and callable(models.__dict__[name]))

# a few command line args
parser = argparse.ArgumentParser(description='Pruning Experiments')
parser.add_argument('-d', dest='dataset', default='cifar10',
                    help='the dataset to use (default: cifar10)')
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='augment dataset with flips and crops')
parser.add_argument('-r', dest='run_cfg', default='A',
                    help='a run configuration from config.py')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg11_bn)')
parser.add_argument('-j', '--workers', default=4, type=int, 
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, 
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('-t', '--test-batch-size', default=256, type=int,
                    help='test mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-ad', '--adam', dest='adam', action='store_true',
                    help='use the adam optimizer')


def main():
    args = parser.parse_args()
    
    # dict for saving results
    results = dict(epoch=[], train=[], test=[], train5 =[], test5 =[], train_loss=[],
                    test_loss=[], p_count=[], avg_pruned_magnitude=[], run_cfg=[])
    if run_cfgs[args.run_cfg].setdefault('pre_prune_acc', False):
        results['pre_prune_acc'] = []
        results['pre_prune_loss'] = []
    if run_cfgs[args.run_cfg].setdefault('gen_matrices', False):   
        results['hessian_spectrum'] = []
        results['grad_covariance_trace'] = []
        results['post_perturb_acc'] = []
        results['post_perturb_loss'] = []
        
    # set run configuration parameters using values from config.py, and
    # defaults stated here when config.py doesn't have a value set
    scratch_prune = run_cfgs[args.run_cfg].setdefault('scratch_prune', False)
    prune_frac = run_cfgs[args.run_cfg].setdefault('prune_frac', None)
    if prune_frac is not None and hasattr(prune_frac, "__len__"):
        prune_frac = list(prune_frac)
    prune_start = run_cfgs[args.run_cfg].setdefault('prune_start_epoch', None)
    prune_end = run_cfgs[args.run_cfg].setdefault('prune_end_epoch', None)
    if prune_start is not None and not hasattr(prune_start, "__len__"):
        prune_start = [prune_start]
        prune_end = [prune_end]
    weight_decay = run_cfgs[args.run_cfg].setdefault('weight_decay', 0)
    target_large = run_cfgs[args.run_cfg].setdefault('target_large', True)
    retrain_period = run_cfgs[args.run_cfg].setdefault('retrain_period', 3)
    scoring = run_cfgs[args.run_cfg].setdefault('scoring', 'L2')
    prune_random = run_cfgs[args.run_cfg].setdefault('prune_random', False)
    run_cfgs[args.run_cfg]['lr_schedule'] = run_cfgs[args.run_cfg].setdefault(
                                        'lr_schedule', [150,300])
    noise = run_cfgs[args.run_cfg].setdefault('noise', None)
    start_epoch = run_cfgs[args.run_cfg].setdefault('start_epoch', 0)
    use_host_name = run_cfgs[args.run_cfg].setdefault('use_host_name', False)
    if run_cfgs[args.run_cfg].setdefault('batch_size', None) is not None:
        args.batch_size = run_cfgs[args.run_cfg].setdefault('batch_size', None)
   
    # the epochs to prune on (formatted as parameters for the range() function
    # for each layer)
    if prune_start is not None:
        prune_epochs = [[s, e + 1, r] for s,e,r in zip(
                    prune_start,prune_end,[retrain_period]*len(prune_start))]

    # create model
    if args.dataset in ["cifar10"]:
        num_classes = 10
    if args.dataset in ["cifar100"]:
        num_classes = 100
    print("=> creating model '{}'".format(args.arch))
    print("=> cfg: prune frac: {}, prune start : {}, prune end: {}, target large: {},"
          " scoring: {}, weight decay: {}, lr_schedule: {}, lr: {}, num_classes: {},"
          " prune_random: {}, retrain: {}, noise: {}, epochs: {}, adam: {}".format(
            prune_frac, prune_start, prune_end, target_large,  scoring, 
            weight_decay, run_cfgs[args.run_cfg]['lr_schedule'], args.lr, num_classes,
            prune_random, retrain_period, noise, args.epochs, args.adam))
    model = models.__dict__[args.arch](dataset = args.dataset,
                                       prune_frac = prune_frac,
                                       num_classes = num_classes,
                                       target_large = target_large,
                                       scoring = scoring,
                                       prune_random = prune_random,
                                       noise = noise)
    
    # using single GPU, but this is still useful for now because it puts the 
    # data onto the GPU by default (i.e. input.to(device) is not necessary)
    if hasattr(model, 'features'):
        model.features = torch.nn.DataParallel(model.features) 
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    torch.backends.cudnn.benchmark = True

    # Data loading code
    train_loader, val_loader, small_loader, batch_size_1_loader = get_train_test_data(args)

    # define loss function (criterion) and pptimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer, lr_scheduler = get_optim_and_scheduler(model.parameters(), args, run_cfgs)


    #####################################
    #####################################
    # begin main training/analysis loop #
    #####################################
    #####################################
    for epoch in range(start_epoch, args.epochs):   
            
        # train for one epoch unless we're doing scratch pruning, then skip
        # the first epoch of training and prune 
        if scratch_prune and start_epoch==epoch: 
            train_acc, train_acc5, train_loss = torch.tensor([0.]), torch.tensor(
                                                                        [0.]), 0
        else:
            train_acc, train_acc5, train_loss = train(train_loader, model, 
                                                      criterion, optimizer, epoch)
        
        model.eval()
        
        # compute test accuracy before pruning
        if run_cfgs[args.run_cfg].setdefault('pre_prune_acc', False):
            print('\npre pruning accuracy...')
            # evaluate on validation set
            pre_prec1, _, pre_loss = validate(val_loader, model, criterion, optimizer) 
            results['pre_prune_acc'].append(float(pre_prec1.cpu()))
            results['pre_prune_loss'].append(pre_loss)
            
        # compute hessian eigenvalues, epsilon flatness, and grad covariance trace
        if run_cfgs[args.run_cfg].setdefault('gen_matrices', False):
            post_perturb_acc = {}
            post_perturb_loss = {}
            trace = 0
            eigenvals = np.zeros(1)
            if epoch == 314:
                trace = covariance_of_gradient_trace(batch_size_1_loader, model,
                                        criterion, optimizer)
                print(trace)
                eigenvals, eigenvecs = compute_hessian_eigenthings(model, small_loader, 
                                        criterion, 100, max_samples = 512)
                print(eigenvals)             
                #perturb the weights along the dominant eigenvector of the hessian
                torch.save({'model_state_dict': model.state_dict()},
                                   socket.gethostname() + "_" + "temp_weights")
                checkpoint = torch.load(socket.gethostname() + "_" + "temp_weights")
                dev = [x for x in model.parameters()][0].device
                epsilons = [round(x,2) for x in np.linspace(-.4,.4,81)]
                for i, eigenvec in enumerate(torch.tensor(eigenvecs.copy()[:10], device = dev)):
                    for e in epsilons:
                        index = 0
                        for p in model.parameters():
                            n = p.numel()
                            p.data += e * eigenvec[index:index+n].reshape_as(p)
                            index += n
                        assert index == eigenvec.numel()
                        post_perturb_acc[i,e], _, post_perturb_loss[i,e] = validate(
                                        val_loader, model, criterion, optimizer)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
            results['hessian_spectrum'].append(np.array(eigenvals))
            results['grad_covariance_trace'].append(trace)
            results['post_perturb_acc'].append(post_perturb_acc)
            results['post_perturb_loss'].append(post_perturb_loss)
            
        # prune the model 
        avg_pruned_magnitude = -1
        if prune_start is not None:
            iters, n_iters = pruning_scheduler(prune_epochs, epoch)
            if type(model) is not torch.nn.parallel.DataParallel:
                avg_pruned_magnitude = model.prune(iters, n_iters)
            else:
                avg_pruned_magnitude = model.module.prune(iters, n_iters)
                
        if not (scratch_prune and start_epoch==epoch):
            lr_scheduler.step()
            
        p_count = pruned_filters(model)
        print("Pruning progress: {} filters/neurons pruned".format(p_count))

        '''
        set pruned weights to zero before post-pruning acc calculations to show that
        custom forward pass is not needed at this point... i.e., test accuracy
        computed after this step is in line with previous epochs' accuracies
        '''
        if epoch==args.epochs-1:
            for layer in [x for x in model.modules() if hasattr(x, 'prune_frac')]:
                currently_pruned = ~layer.mask.bool()
                layer.weight.data[currently_pruned] *= 0
        
        # evaluate on validation set *after* pruning
        print('\npost pruning accuracy...')
        prec1, prec5, test_loss = validate(val_loader, model, criterion, optimizer) 
        
        #update dict
        results['epoch'].append(epoch+1)
        results['train'].append(float(train_acc.cpu()))
        results['train_loss'].append(train_loss)
        results['test'].append(float(prec1.cpu()))
        results['test_loss'].append(test_loss)
        results['train5'].append(float(train_acc5.cpu()))
        results['test5'].append(float(prec5.cpu()))
        results['p_count'].append(p_count)
        results['avg_pruned_magnitude'].append(avg_pruned_magnitude if 
                                       avg_pruned_magnitude>=0 else None)
        results['run_cfg'].append(args.run_cfg)

    #####################################
    #####################################
    #  end main training/analysis loop  #
    #####################################
    #####################################
        
           
    # write out results dict as a pickled dataframe
    file = ("logs/" + args.dataset + "_" + args.arch + "_" + args.run_cfg.split('_')[0])
    if os.path.isfile(file):
        df = pd.read_pickle(file)
        df = df.append(pd.DataFrame(results), ignore_index=True)
    else:
        df = pd.DataFrame(results)
    df.to_pickle(file)


if __name__ == '__main__':
    main()
