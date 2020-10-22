from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


data = preprocess('logs/cifar100_resnet18_TF1')
data = add_pruning_status_vars(data)
data = add_stability_vars(data)
summary = summarize(data)

styles = {'TF1_0': 'No Pruning', 
          'TF1_3':'Scratch 15%',
          'TF1_2':'Prune_S 15%',
          'TF1_1':'Prune_L 15%'}
summary['Pruning Style'] = [styles[r] for r in summary.run_cfg]

print(summary.groupby('Pruning Style')['Best Test Accuracy in Run'].mean())
print(summary.groupby('Pruning Style')['Best Test Accuracy in Run'].std())
print(summary.groupby('Pruning Style')['Mean Stability (%)'].mean())
print(summary.groupby('Pruning Style')['Mean Stability (%)'].std())

print(stats.ttest_ind(summary.query("run_cfg=='TF1_1'")['Best Test Accuracy in Run'].values,
                summary.query("run_cfg=='TF1_2'")['Best Test Accuracy in Run'].values))
print(stats.ttest_ind(summary.query("run_cfg=='TF1_1'")['Best Test Accuracy in Run'].values,
                summary.query("run_cfg=='TF1_3'")['Best Test Accuracy in Run'].values))