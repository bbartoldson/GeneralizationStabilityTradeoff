import pandas as pd
import numpy as np


def summarize(f, other_group_vars = None):
    '''
    compute best test accuracy (and gen. gap) in run, mean stability during run.
    return a dataframe with one row per run and these added stats

    Parameters
    ----------
    f : pandas.core.frame.DataFrame
        a loaded log file.

    Returns
    -------
    summary : pandas.core.frame.DataFrame
        a summarized version of the input log file

    '''
    
    if other_group_vars == None:
        other_group_vars = []
    mean_stability = f.query("pruned_epoch==True").groupby(
                        ['run','run_cfg',*other_group_vars])['Stability'].mean().reset_index()
    best_acc = f.query("post_prune").groupby(
                        ['run_cfg','run',*other_group_vars])['test'].max().reset_index()
    summary = pd.merge(best_acc, mean_stability, how='left') 
    
    # scratch pruning and no pruning have 100% pruning stability during training
    summary.loc[summary.Stability.isnull(),'Stability'] = 100
    
    # add generalization gap from the best performing epoch
    best = f.query("post_prune").groupby(['run','run_cfg',*other_group_vars]
                                            )['test'].idxmax().reset_index().test
    best_epochs = f.iloc[best]
    best_epochs['Generalization Gap'] = best_epochs.train-best_epochs.test
    gap_summary = best_epochs.groupby(['run','run_cfg']
                                      )['Generalization Gap'].min().reset_index()
    summary = pd.merge(summary, gap_summary)
    
    summary.rename(columns={'test':'Best Test Accuracy in Run'},inplace=True)
    summary.rename(columns={'Stability':'Mean Stability (%)'},inplace=True)  
    
    return summary


def add_pruning_status_vars(f):    
    '''
    add vars indicating:
        an epoch on which pruning occurred, 
        the last epoch on which pruning will occur for the cfg,
        whether pruning has ended and it's thus safe to measure test accuracy
            (you can't measure a cfg's test accuracy until pruning is over)

    Parameters
    ----------
    f : pandas.core.frame.DataFrame
        a loaded log file.

    Returns
    -------
    f : pandas.core.frame.DataFrame
        a loaded log file with additional variables.

    '''
    
    # cfg - epoch combinations that had pruning
    pruned_epoch = f[f.avg_pruned_magnitude>0][['run_cfg','epoch']].drop_duplicates()
    
    # last epoch on which pruning occurred for each cfg
    last_pruned_epoch = pruned_epoch.groupby('run_cfg').max().reset_index()
    
    pruned_epoch['pruned_epoch'] = True
    last_pruned_epoch.rename(columns={'epoch':'last_pruned_epoch'},inplace=True)   
    f = pd.merge(f, pruned_epoch, how='left')
    f = pd.merge(f, last_pruned_epoch, how='left')    
    # if never pruned or pruned at start of training (on epoch 1 but before
    # training starts, the "scratch" method), set to 0
    f.loc[~(f.last_pruned_epoch>1),'last_pruned_epoch'] = 0
    f.loc[f.last_pruned_epoch==0,'pruned_epoch'] = False
    f.loc[f.pruned_epoch.isnull(), 'pruned_epoch'] = False
    
    f['post_prune'] = f.epoch>=f.last_pruned_epoch
    
    return f


def add_stability_vars(f):
    '''
    adds stability and instability variables to a loaded log file

    Parameters
    ----------
    f : pandas.core.frame.DataFrame
        a loaded log file.

    Returns
    -------
    f : pandas.core.frame.DataFrame
        a loaded log file with additional variables.

    '''
    f['Instability'] =  (f.pre_prune_acc - f.test) / f.pre_prune_acc
    # compute stability as a percentage
    f['Stability'] = 100 * (1 - f['Instability'])   
    return f


def preprocess(path):
    '''    
    load a pickled log file and add a unique run number to each run
    
    Parameters
    ----------
    path : str
        a path to a pickled dataframe containing variables saved on each epoch
        of each run (a log file). has runs x epochs-per-run rows

    Returns
    -------
    data : pandas.core.frame.DataFrame
        the loaded pandas dataframe with a run number added

    '''
    # load
    data = pd.read_pickle(path)
    
    #add run
    run = np.zeros(len(data))
    for i, (epoch, prior_epoch) in enumerate(zip(data.epoch[1:], data.epoch[:-1])):
        if epoch<prior_epoch:
            run[i+1:] +=1
    data['run'] = pd.Series(run, index=data.index)
    
    return data


def stack(f1, f2):
    '''
    append f2 to f1
        
        note that the run number of f2 is increased by f1.run.max()+1 to
        ensure run number is unique in the returned dataframe
    
    Parameters
    ----------
    f1 : pandas.core.frame.DataFrame
        a dataframe with runs x epochs-per-run rows
    f2 : pandas.core.frame.DataFrame
        a dataframe with runs x epochs-per-run rows

    Returns
    -------
    stacked : pandas.core.frame.DataFrame
        a dataframe with f2 appended to the end of f1.
        if f1 has n rows and f2 has m rows, stacked will have n+m rows

    '''
    if f2.run.min() <= f1.run.max():
        f2.run = f2.run+f1.run.max()+1
    stacked = f1.append(f2,ignore_index=True,verify_integrity=True)
    return stacked
    

def calc_n_batches_of_training_noise(x, test_batch_size):
    '''
    a function to compute how many training batches had weights noised.
    see example below with calc(3000) returning 2480.
    
    with a 256 test batch size, testing takes 40 batches
    training takes 391 with a 128 batch size
    
    batches_since_targeting (see models.py) is updated on train and test batches, 
    so to find how many training batches had the weights zeroed, use this formula
    combined with the batches_of_noise specified in the config's noise variable.

    if x=0, this will return 0 but actually there is 1 batch of noise, see
    models.py.
    '''
    n = 0
    def test(x):
        return x-test_batch_size
    def train(x,n):
        n += min(x,391)
        return x-391, n
    x=test(x)
    while x>0:
        x, n=train(x,n)
        x=test(x)
        x=test(x)
    return n