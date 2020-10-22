from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


data = preprocess("logs/cifar10_resnet18_FC3")
data = add_pruning_status_vars(data)
data = add_stability_vars(data)
summary = summarize(data)


styles = {'FC3_0':'Prune_S 10% Total',
          'FC3_1':'Prune_S 6% Total',
          'FC3_2':'No Pruning' }
summary['Pruning Detail'] = [styles[r] for r in summary.run_cfg]
stable_dict = dict([(k, (str(round(v,2)) if v!=100 else '100') + '% Stable (' + k + ')')
      for k, v in summary.groupby("Pruning Detail")['Mean Stability (%)'].mean().items()])
summary['Stability (Method)'] = [stable_dict[k] for k in summary['Pruning Detail']]

print(summary.groupby('Stability (Method)')['Best Test Accuracy in Run'].mean())
print(summary.groupby('Stability (Method)')['Best Test Accuracy in Run'].var())
print(summary.groupby('Stability (Method)')['Mean Stability (%)'].mean())
print(summary.groupby('Stability (Method)')['Mean Stability (%)'].var())


summary = summary.query("run_cfg!='FC3_2'")
r_palette ={'FC3_0':sns.color_palette(n_colors = 6)[0],
            'FC3_1':sns.color_palette(n_colors = 6)[1]
                        }

plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':24})
g = sns.JointGrid(data=summary, x='Mean Stability (%)', 
                  y='Best Test Accuracy in Run', ratio=100)
g.plot_joint(sns.scatterplot,s=0)
g.annotate(stats.kendalltau) 
kendall = g.ax_joint.get_children()[-2]
g.annotate(stats.pearsonr) 
a=g.ax_joint.get_children()[-2]
a.set_zorder(5)
a.draw_frame(0)
a.set_visible(False)
sns.regplot(data=summary,x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), scatter=False, ci=None, color='grey')
sns.scatterplot(data=summary,x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), hue='run_cfg', palette=r_palette,
            legend =False, s=82)
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gcf().set_size_inches(8, 6)
plt.annotate(a.get_texts()[0].get_text(), (99.86,93.8))
plt.annotate(kendall.get_texts()[0].get_text(), (99.86,93.6))
g.savefig('figures_and_tables/PDFs/Figure_C3_Gen.pdf')


plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':24})
g = sns.JointGrid(data=summary, x='Mean Stability (%)',
                  y='Generalization Gap', ratio=100)
g.plot_joint(sns.scatterplot,s=0)
g.annotate(stats.kendalltau) 
kendall = g.ax_joint.get_children()[-2]
g.annotate(stats.pearsonr) 
a=g.ax_joint.get_children()[-2]
a.set_zorder(5)
a.draw_frame(0)
a.set_visible(False)
sns.regplot(data=summary,x='Mean Stability (%)', y='Generalization Gap',
            ax = plt.gca(), scatter=False, ci=None, color='grey')
sns.scatterplot(data=summary,x='Mean Stability (%)', y='Generalization Gap',
            ax = plt.gca(), hue='run_cfg', palette=r_palette,
            legend =False, s=82)
plt.gca().set_ylabel('Generalization Gap (%)')
plt.gcf().set_size_inches(8, 6)
plt.annotate(a.get_texts()[0].get_text(), (99.86,6.6))
plt.annotate(kendall.get_texts()[0].get_text(), (99.86,6.4))
g.savefig('figures_and_tables/PDFs/Figure_C3_Gap.pdf')