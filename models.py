'''
builds models with functions to enable pruning / temporary noising of params
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from types import MethodType
from math import ceil, floor
from architectures_and_layers.custom_batch_norm import _BatchNorm 
from architectures_and_layers import resnet2056_arch, resnet18_arch, vgg_arch
from scipy.stats import truncnorm
from numpy.random import permutation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_prunable_batch_forward = _BatchNorm._prunable_batch_forward # batch norm with a mask for pruning
         
        
###\############################
# code to make models prunable #
################################

def _prune(self, iterations, N_iters):     
    '''
    prune each prunable layer in the network
    '''  
    avg_pruned_magnitude = 0
    i = 0
    for layer in [x for x in self.modules() if hasattr(x, 'prune_frac')]:  
        iteration = iterations.pop(0)
        N_iter = N_iters.pop(0)
        if iteration:          
            temp = layer.prune_layer(iteration, N_iter)
            if np.isnan(temp):
                print('nan pruned mag!!')
                continue
            else:
                assert temp > -1e-5, 'negative pruned magnitude!!'
            avg_pruned_magnitude += max(temp,0)
            i+=1
    if i==0:
        return -1 # magnitude calculated is always positive, so negative means no pruning
    return avg_pruned_magnitude / i    


def _prunable_conv_forward(self, input):
    '''
    modify the forward pass to mask pruned filters
    '''
    return F.conv2d(input, self.weight.transpose(0,3).mul(self.mask).transpose(0,3),
                    self.bias, self.stride, self.padding, self.dilation, self.groups)


def _noisy_conv_forward(self, input):
    '''
    modify the forward pass to add noise to targeted filters
    '''
    # add noise to weights for self.batches_of_noise batches.
    # (this will add noise during test batches too, which creates a relatively large
    # change in the weights when gaussian noise is added because no training is
    # offsetting the effect)
    if self.batches_since_targeting <= self.batches_of_noise:
        self.batches_since_targeting += 1
        w = self.weight.data
        unpruned_filter_std = w[self.mask.bool()][0].std().item()
        if 'restore' in self.noise:
            bn_w = self.next_bn_layer.weight.data
            bn_b = self.next_bn_layer.bias.data
            bn_mu = self.next_bn_layer.running_mean.data
            bn_sig2 = self.next_bn_layer.running_var.data
        if self.noise[:4] == 'iter':
            if self.batches_since_targeting%900 == 1:
                print('targeting newest weights')
            pruned = self.newly_pruned
        else:
            if self.batches_since_targeting%900 == 1:
                print('targeting all weights (cumulative)')
            pruned = ~self.mask.bool()
            
        if self.batches_since_targeting == 1 and 'restore' in self.noise:
            self.orig = w[pruned]
            self.orig_bn_weight = bn_w[pruned]
            self.orig_bn_bias = bn_b[pruned]
            self.orig_bn_mu = bn_mu[pruned]
            self.orig_bn_sig2 = bn_sig2[pruned]
        
        if 'gaussian' in self.noise:
            if self.batches_since_targeting%900 == 1:
                print('adding gaussian noise')
            w[pruned] += torch.distributions.Normal(0, unpruned_filter_std
                                            ).sample(w[pruned].shape).cuda()
        else:
            if self.batches_since_targeting%900 == 1:
                print('multiplying zeroing noise')
            w[pruned] *= 0
            if 'restore' in self.noise:
                bn_w[pruned] *= 0
                bn_b[pruned] *= 0
    
        if (self.batches_since_targeting > self.batches_of_noise)  and ('restore'
                                                        in self.noise):
            w[pruned] = self.orig
            bn_w[pruned] = self.orig_bn_weight
            bn_b[pruned] = self.orig_bn_bias
            bn_mu[pruned] = self.orig_bn_mu
            bn_sig2[pruned] = self.orig_bn_sig2
            
    return F.conv2d(input, self.weight,
                    self.bias, self.stride, self.padding, self.dilation, self.groups)
    
            
def _make_conv_prunable(layer, prune_frac, target_large, prune_random=False,
                        next_bn_layer = None, noise = None, scoring = 'L2'):    
    '''
    modify a conv layer to make it prunable with an l1 or l2 norm score,
    
    note the following BN layer will be masked wherever a filter in this
    conv layer is masked
    '''
    assert type(layer) is nn.modules.conv.Conv2d, print(type(layer))
    layer.prune_layer = MethodType(_prune_layer, layer)
    # the pruning mask
    layer.mask = torch.ones_like(layer.weight.transpose(0,3)[0,0,0], device=device)
    # a big number that won't be reached, used in noise experiments
    layer.batches_since_targeting = 1e7 
    # if applying noise, the type (Gaussian or zeroing)
    layer.noise = noise
    layer.forward = MethodType(_prunable_conv_forward, layer)
    if layer.noise:
        if not ('Prune' in layer.noise):
            # test after initial prune, then train, so 79+391 batches per epoch.
            # (assuming a batch size of 128 and 50k train + 10k test).
            # we apply noise for batches_of_noise batches
            layer.batches_of_noise = int(layer.noise.split("_")[1]) 
            layer.forward = MethodType(_noisy_conv_forward, layer)
    # fraction of weights to remove
    layer.prune_frac = prune_frac 
    # True: prune largest weights, False: smallest
    layer.target_large = target_large
    # True: prune random weights and override target_large
    layer.prune_random = prune_random
    layer.scoring = scoring
    if layer.noise == None: 
        # only mask the next BN layer if a noise experiment is not being run
        # because we test the effect of applying noise to the conv weights
        next_bn_layer.mask = layer.mask
        next_bn_layer.forward = MethodType(_prunable_batch_forward, next_bn_layer)
    else:
        layer.next_bn_layer = next_bn_layer
    if 'activations' in layer.scoring:
        layer.post_shortcut_running_mean = torch.tensor([0.],device=device)
        layer.counter = 0 # only update this running mean every 10 batches
        
    
def _make_bn_prunable(layer, prune_frac, target_large, prune_random=False,
                      scoring = 'EBN'):    
    '''
    modify a batch norm layer to make it prunable with the E[BN] approach from our appendix
    '''
    assert type(layer) in (nn.modules.batchnorm.BatchNorm2d,nn.modules.batchnorm.BatchNorm1d
                          ), print(type(layer))
    layer.prune_layer = MethodType(_prune_layer, layer)
    # the pruning mask
    layer.mask = torch.ones_like(layer.weight, device=device)
    layer.forward = MethodType(_prunable_batch_forward, layer)
    # fraction of weights to remove
    layer.prune_frac = prune_frac 
    # True: prune largest weights, False: smallest
    layer.target_large = target_large      
    # True: prune random weights and override target_large
    layer.prune_random = prune_random
    layer.post_relu_running_mean = torch.tensor([0.], device = device)
    layer.scoring = scoring
    
    def _bn_expectation(self):
        '''
        computes expected value of post-ReLU, batch-normalized feature-map-activations
        
        if 'alt' is in the scoring method, then will use a variant of the E[BN] method
        that provided more stability in our ResNet18 experiments
        '''
        def _normalize(x, mu, std):
            return (x-mu)/std    
        # alpha is the point in the std normal dist corresponding to 0 (the 
        # ReLU-truncation point) in the BN dist
        if 'alt' in layer.scoring:
            w = self.weight.detach().to(device)
        else:
            w = self.weight.detach().abs().to(device)
        alpha = _normalize(0, self.bias.detach().to(device), w)
        n = torch.distributions.normal.Normal(torch.tensor([0.], device=device),
                                              torch.tensor([1.], device=device))
        # Z is the area under the curve retained in the normalized analog of the
        # ReLU-truncated dist
        Z = 1 - n.cdf(alpha)
        # the expected value of the truncated distribution (what makes it past the ReLU)
        trunc_expected = self.bias.detach().to(device) + n.log_prob(alpha).exp() / Z * w
        # if Z=0, alpha=inf, weight=0, bias < 0
        trunc_expected[Z==0] = 0 
        # the expected value of batch-normalized, post-ReLU feature maps is
        # n.cdf(alpha)*0 + (1-n.cdf(alpha))*trunc_expected
        self.expectation = Z * trunc_expected
        # Z=nan implies alpha = 0/0, so bias and weight are 0
        self.expectation[torch.isnan(Z)] = 0 
        
    layer.calc_expectation = MethodType(_bn_expectation, layer)
            

def _prune_layer(self, iteration, N_iter):
    
    noise = None
    if hasattr(self, 'noise'):
        noise = self.noise

    if iteration == 1:               
        self.N_filters = self.weight.size(0) 
        self.final_kept_filters = ceil((1-self.prune_frac) * self.N_filters)                         
        
    # calculate the number of filters (k) to prune this iteration
    self.current_kept_filters = ceil(self.final_kept_filters + 
        (self.N_filters - self.final_kept_filters) * (N_iter-iteration)/(N_iter))
    pruned_count = (~self.mask.bool()).sum()
    k = int(self.N_filters - self.current_kept_filters - pruned_count) 
    
    # score pruning targets
    if 'L1' in self.scoring:
        print('pruning with filter L1 norm...')
        target = self.weight.data[self.mask.bool()].view(self.mask.int().sum(), -1).norm(p=1,dim=1)
    elif 'EBN' in self.scoring:
        print('pruning with E[BN]')
        self.calc_expectation()
        target = self.expectation.data[self.mask.bool()]
    elif 'activations' in self.scoring:
        print("pruning with post-shortcut activations' L1 norm")
        target = self.post_shortcut_running_mean.data[self.mask.bool()]
    else:
        print('pruning with filter L2 norm')
        target = self.weight.data[self.mask.bool()].view(self.mask.int().sum(), -1).norm(dim=1)      
        
    # prune if k>0
    if k > 0:    
        k_weights, _ = torch.topk(target, k, largest=self.target_large)
        cutoff_weight = float(k_weights[-1])
        if self.target_large:
            meets_cutoff = target < cutoff_weight
        else:
            meets_cutoff = target > cutoff_weight
        if (~meets_cutoff).sum() == len(meets_cutoff):
            print("you're trying to prune the whole layer, no variance in "+
                  "weights is likely. proceeding by pruning randomly")
            meets_cutoff[k:] = True
        if (~meets_cutoff).sum() > k:
            print("you're trying to prune too much, sparsity from weight decay "+
                  "is likely. proceeding by pruning only the scheduled # of weights")
            temp = meets_cutoff[~meets_cutoff]
            temp[k:] = True
            meets_cutoff[~meets_cutoff] = temp
        assert (~meets_cutoff).sum() == k, 'pruned fewer weights than should have!!!'
        if self.prune_random:
            meets_cutoff = meets_cutoff[permutation(range(len(meets_cutoff)))]
        #magnitude of cut weights
        avg_pruned_magnitude = float(target[~meets_cutoff].mean())
        print('avg score of pruned weights was {}'.format(avg_pruned_magnitude))
        
        if noise != None:
            if not ('Prune' in self.noise):
                self.batches_since_targeting = 0
                temp = self.mask.data.clone()
                temp[self.mask.bool()] += (~meets_cutoff).float()
                self.newly_pruned = temp==2
                                    
        self.mask.data[self.mask.bool()] = meets_cutoff.float()
        
        if noise == None:    
            self.weight.data[~self.mask.bool()] *= 0                
                
        return avg_pruned_magnitude
    
    return float('nan')
            
    
    
#############################
######## models #############
#############################
    
def resnetxx(dataset = "cifar10", 
          prune_frac = None, 
          target_large = True,
          prune_random=False,
          scoring = 'L2',
          layers = None):
    '''
    set up resnet20 or resnet56 for pruning. 
    
    we only prune layers that are in the blocks.
    
    shortcut connections to a pruned filter get pruned via logic in
    _mask_shortcuts_forward_C() function
    '''
    if layers == 56:
        model = resnet2056_arch.resnet56() #assume cifar10
    else:
        assert layers == 20
        model = resnet2056_arch.resnet20() #assume cifar10     
    
    def _mask_shortcuts_forward_C(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)    
        # instead of the prior line, use this to prevent shortcuts to pruned filters
        out[:,self.conv2.mask.bool()] += self.shortcut(x)[:,self.conv2.mask.bool()] 
        out = F.relu(out)
        return out
    
    if prune_frac is not None:
        mods = [*model.layer1.modules()] + [*model.layer2.modules()] + [*model.layer3.modules()]
        bn_indices = list(np.nonzero([type(x)==torch.nn.modules.batchnorm.BatchNorm2d 
                                                  for x in mods])[0][::-1])
        conv_indices = list(np.nonzero([type(x)==torch.nn.modules.conv.Conv2d 
                                                  for x in mods])[0][::-1])
        assert len(bn_indices) == len(prune_frac)
        assert (np.array(bn_indices) - np.array(conv_indices) == 1).all() 
        prune_frac_len = len(prune_frac)
        prunable_layers = len(bn_indices)
        model.targeted_bn_layer_indices = bn_indices[:prune_frac_len]
        for layer in model.targeted_bn_layer_indices:
            _make_conv_prunable(mods[layer-1], prune_frac.pop(), target_large, 
                    prune_random, next_bn_layer = mods[layer], scoring=scoring)    
        blocks = list(model.layer1) + list(model.layer2) + list(model.layer3)
        for block in blocks:
            block.forward = MethodType(_mask_shortcuts_forward_C, block) 
        assert len(prune_frac)==0
        model.prune = MethodType(_prune, model)
    return model   


def resnet20(dataset = "cifar10", 
          prune_frac = None, 
          num_classes = 10, 
          target_large = True,
          scoring = 'L2',
          prune_random=False,
          noise = None):
    # only running cifar10 experiments with this model
    assert num_classes == 10
    # didn't test adding noise
    assert noise == None
    return resnetxx(dataset = dataset, 
          prune_frac = prune_frac, 
          target_large = target_large,
          prune_random=prune_random,
          scoring = scoring,
          layers=20)


def resnet56(dataset = "cifar10", 
          prune_frac = None, 
          num_classes = 10, 
          target_large = True,
          scoring = 'L2',
          prune_random=False,
          noise = None):
    # only running cifar10 experiments with this model
    assert num_classes == 10
    # didn't test adding noise
    assert noise == None
    return resnetxx(dataset = dataset, 
          prune_frac = prune_frac, 
          target_large = target_large,
          prune_random=prune_random,
          scoring = scoring,
          layers=56)
    

def resnet18(dataset = "cifar10", 
          prune_frac = None, 
          num_classes = 10, 
          target_large = True,
          scoring = 'L2',
          prune_random=False,
          noise = None):
    '''
    set up resnet18 for pruning. 
    
    we only prune layers that are in the blocks.
    
    shortcut connections to a pruned filter get pruned via logic in
    _mask_shortcuts_forward_C or _mask_shortcuts_forward_B (the latter
    is for our custom E[BN] pruning approach)
    '''
    model = resnet18_arch.ResNet18(num_classes=num_classes)
    def _mask_shortcuts_forward_B(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)    
        # instead of the prior line, use this to prevent shortcuts to pruned filters
        out[:,self.bn2.mask.bool()] += self.shortcut(x)[:,self.bn2.mask.bool()] 
        out = F.relu(out)
        return out
    def _mask_shortcuts_forward_C(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)    
        # instead of the prior line, use this to prevent shortcuts to pruned filters
        out[:,self.conv2.mask.bool()] += self.shortcut(x)[:,self.conv2.mask.bool()]
        out = F.relu(out)
        return out
    if 'activations' in scoring:
        def _mask_shortcuts_forward_C(self, x):        
            out = self.bn1(self.conv1(x))
            # only compute the mean if using training data (every 10 batches)
            if self.training:
                self.conv1.counter+=1
                if self.conv1.counter%10==0: 
                    self.conv1.post_shortcut_running_mean = .1*out.abs().sum(
                        dim=[0,2,3]).detach() + .9*self.conv1.post_shortcut_running_mean.detach()        
            out = F.relu(out)
            out = self.bn2(self.conv2(out))
            #out += self.shortcut(x)    
            # instead of the prior line, use this to prevent shortcuts to pruned filters
            out[:,self.conv2.mask.bool()] += self.shortcut(x)[:,self.conv2.mask.bool()] 
            if self.training and self.conv1.counter%10==0:
                self.conv2.post_shortcut_running_mean = .1*out.abs().sum(
                       dim=[0,2,3]).detach() + .9*self.conv2.post_shortcut_running_mean.detach()
            out = F.relu(out)
            return out
        
    if prune_frac is not None:
        mods = [*model.layer1.modules()] + [*model.layer2.modules()] + [
                        *model.layer3.modules()] + [*model.layer4.modules()] 
        bn_indices = list(np.nonzero([type(x)==torch.nn.modules.batchnorm.BatchNorm2d 
                                                  for x in mods])[0][::-1])
        conv_indices = list(np.nonzero([type(x)==torch.nn.modules.conv.Conv2d 
                                                for x in mods])[0][::-1])
        # shortcut conv layers don't need their own mask,
        # we prune them by using the masks of the filters they add to
        for i in range(len(conv_indices)):
            if mods[conv_indices[i]].weight.shape[-2:] == torch.Size([1,1]):
                bn_indices[i] = conv_indices[i] = 999
        for i in range(3):
            bn_indices.remove(999)
            conv_indices.remove(999)
        assert (np.array(bn_indices) - np.array(conv_indices) == 1).all() 
        prune_frac_len = len(prune_frac)
        prunable_layers = len(bn_indices)
        model.targeted_bn_layer_indices = bn_indices[:prune_frac_len]
        for layer in model.targeted_bn_layer_indices:
            if 'EBN' in scoring:
                _make_bn_prunable(mods[layer], prune_frac.pop(), target_large, prune_random,
                                  scoring=scoring)
            else:
                _make_conv_prunable(mods[layer-1], prune_frac.pop(), target_large, 
                            prune_random, next_bn_layer = mods[layer], scoring=scoring)    
        pruned_blocks = list(model.layer4)
        if prune_frac_len > 4: 
            # we've given a fraction for multiple blocks
           pruned_blocks +=  list(model.layer3) + list(model.layer2) + list(model.layer1)
        for block in pruned_blocks:
            if 'EBN' in scoring:
                block.forward = MethodType(_mask_shortcuts_forward_B, block) 
            else:
                block.forward = MethodType(_mask_shortcuts_forward_C, block) 
        assert len(prune_frac)==0
        model.prune = MethodType(_prune, model)
    return model   
    
    
def vgg11_bn(dataset = "cifar10", 
          prune_frac = None, 
          num_classes = 10, 
          target_large = True,
          scoring = 'L2',
          prune_random=False,
          noise = None):
    model = vgg_arch.vgg11_bn(num_classes = num_classes)
    if dataset in ["cifar10", "mnist", "cifar100"]:
        model.classifier = nn.Sequential(nn.Linear(512, num_classes))
    if prune_frac is not None:
        model.targeted_bn_layer_indices = [-3,-6,-10,-13]
        for layer in model.targeted_bn_layer_indices:
            if 'EBN' in scoring:
                _make_bn_prunable(model.features[layer], prune_frac.pop(), target_large,
                                  prune_random, scoring=scoring)
            else:
                _make_conv_prunable(model.features[layer-1], prune_frac.pop(), target_large,
                  prune_random, next_bn_layer = model.features[layer], noise=noise, scoring=scoring)
        assert len(prune_frac)==0
        model.prune = MethodType(_prune, model)
    return model   