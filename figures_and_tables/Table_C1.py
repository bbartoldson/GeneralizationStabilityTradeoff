from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


####################
# preprocess data #
###################

data = preprocess("logs/cifar10_resnet18_TC1")
data = add_stability_vars(data)
data = add_pruning_status_vars(data)
summary = summarize(data)

# define a mapping between configurations and method details (iterative rate and
# pruning target) for legend labeling
method_dict = {'TC1_1': 'Prune_S 1%',
       'TC1_2': 'Prune_S 13%',
       'TC1_3': 'Prune_L 13%',}

summary['Method'] = [method_dict[k] for k in summary.run_cfg]

print('mean: \n', summary.groupby('run_cfg')[['Best Test Accuracy in Run',
                                             'Mean Stability (%)']].mean())
print('\nstd: \n', summary.groupby('run_cfg')[['Best Test Accuracy in Run',
                                             'Mean Stability (%)']].std())
