from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


###################
###################
# First plot VGG #
###################
###################


####################
# preprocess data #
###################

data = preprocess("logs/cifar10_vgg11_bn_FB2")
baseline = preprocess("logs/cifar10_vgg11_bn_F2").query("run_cfg=='F2_VGG_0'")
data = stack(data,baseline)
data = add_stability_vars(data)
data = add_pruning_status_vars(data)
summary = summarize(data)

###############################################
#      plot legend labeling and coloring      #
###############################################

# define a mapping between configurations and method details (iterative rate and
# pruning target) for legend labeling
method_dict = {'F2_VGG_0': 'No Pruning',
       'FB2_VGG_1': 'Prune_S 1%',
       'FB2_VGG_2': 'Prune_S 13%',
       'FB2_VGG_3': 'Prune_L 13%'}

# a mapping from method to color plotting
color_dict = {'No Pruning': sns.color_palette(n_colors = 6)[0],
       'Prune_S 1%': sns.color_palette(n_colors = 6)[1],
        'Prune_S 13%': sns.color_palette(n_colors = 6)[2],
        'Prune_L 13%': sns.color_palette(n_colors = 6)[3]}

summary['Method'] = [method_dict[k] for k in summary.run_cfg]
# a mapping from method to stability+method (again for legend labeling)
stability_dict = dict([(k, (str(round(v,2)) if v!=100 else '100') + '% Stable (' 
                     + k + ')') for k, v in summary.groupby("Method"
                                    )['Mean Stability (%)'].mean().items()])
summary['Stability (Method)'] = [stability_dict[k] for k in summary['Method']]
stability_palette = dict([(v, color_dict[k]) for k, v in stability_dict.items()])


#############
# plotting #
############

# don't show the baseline in the corr plot
summary = summary.query('run_cfg!="F2_VGG_0"')

# correlation plot
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':24})
g = sns.JointGrid(data=summary, 
                  x='Mean Stability (%)',
                  y='Best Test Accuracy in Run', ratio=100)
g.plot_joint(sns.scatterplot, s=0)
g.annotate(stats.kendalltau)
kendall = g.ax_joint.get_children()[-2]
g.annotate(stats.pearsonr)
a=g.ax_joint.get_children()[-2]
a.set_zorder(5)
a.draw_frame(0)
a.set_visible(False)   
sns.regplot(data=summary,
            x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
sns.scatterplot(data=summary,
                x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), hue='Stability (Method)', palette=stability_palette,
            legend = False, s = 82)
plt.annotate(a.get_texts()[0].get_text(), (96.4, 85.5))
plt.annotate(kendall.get_texts()[0].get_text(), (96.4, 85.3))
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gcf().set_size_inches(8, 6)
g.savefig('figures_and_tables/PDFs/Figure_B2_VGG_correlation.pdf')

# dynamics plot
data['Method'] = pd.Series([method_dict[r] for r in data.run_cfg],
                                                         index=data.index)
data['Stability (Method)']  = [stability_dict[r] for r in data['Method'] ]

plt.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 
                         'legend.fancybox':True, 'legend.shadow':True})
plt.rcParams.update({'axes.labelsize':20})
g = sns.FacetGrid(data= data, hue= 'Stability (Method)', 
                  hue_order= [stability_dict[r] for r in
                    ['No Pruning',  'Prune_S 1%', 'Prune_S 13%', 'Prune_L 13%']])
g=(g.map(sns.lineplot, 'epoch', 'test'))
plt.gca().set_ylim(83,86.5)
plt.gca().set_xlim(1,325)
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gca().set_xlabel('Epoch')
plt.gcf().set_size_inches(12, 4.5)
plt.gca().tick_params(labeltop=False, labelright=True)
plt.gcf().tight_layout()
l=g.axes.flatten()[0].legend(loc='upper left', fancybox=True, shadow=True,
            bbox_to_anchor=(.08,.94,0,0),bbox_transform=plt.gcf().transFigure)
l.set_title('Stability (Method)')
plt.setp(l.get_title(), fontsize=15, weight='bold')
red1 = np.array([250,83.5])
plt.gca().annotate('', xy=(170,84.4), xytext=(200,84.4),
            arrowprops=dict(facecolor='red', shrink=0.15), )
plt.gca().annotate('', xy=(210,84), xytext=(240,84),
            arrowprops=dict(facecolor='red', shrink=0.05), )
plt.gca().annotate('\u221229%', xy=(250,83.6), xytext=(280,83.6),va='center',
            arrowprops=dict(facecolor='red', shrink=0.05), )
plt.gca().annotate('', xy=(250,84.4), xytext=(280,84.4),
            arrowprops=dict(facecolor='green', shrink=0.05), )
g.axes[0][0].get_lines()[2].set_zorder(3)
plt.gcf().savefig("figures_and_tables/PDFs/Figure_B2_VGG_dynamics.pdf", 
                  bbox_inches = "tight")




###################
###################
# Now plot ResNet #
###################
###################


####################
# preprocess data #
###################

data = preprocess("logs/cifar10_resnet18_FB2")
baseline = preprocess("logs/cifar10_resnet18_F2").query("run_cfg=='F2_RN_0'")
data = stack(data,baseline)
data = add_stability_vars(data)
data = add_pruning_status_vars(data)
summary = summarize(data)

###############################################
#      plot legend labeling and coloring      #
###############################################

# define a mapping between configurations and method details (iterative rate and
# pruning target) for legend labeling
method_dict = {'F2_RN_0': 'No Pruning', 
          'FB2_RN_1':'Prune_S 1%',
          'FB2_RN_2':'Prune_S 14%',
          'FB2_RN_3':'Prune_L 14%'}
# a mapping from method to color plotting
color_dict = {'No Pruning': sns.color_palette(n_colors = 6)[0],
       'Prune_S 1%': sns.color_palette(n_colors = 6)[1],
        'Prune_S 14%': sns.color_palette(n_colors = 6)[2],
        'Prune_L 14%': sns.color_palette(n_colors = 6)[3]}

summary['Method'] = [method_dict[k] for k in summary.run_cfg]
# a mapping from method to stability+method (again for legend labeling)
stability_dict = dict([(k, (str(round(v,2)) if v!=100 else '100') + '% Stable (' 
                     + k + ')') for k, v in summary.groupby("Method"
                                    )['Mean Stability (%)'].mean().items()])
summary['Stability (Method)'] = [stability_dict[k] for k in summary['Method']]
stability_palette = dict([(v, color_dict[k]) for k, v in stability_dict.items()])


#############
# plotting #
############

# remove baseline for corr plot
summary = summary.query('run_cfg!="F2_RN_0"')

# correlation plot
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':24})
g = sns.JointGrid(data=summary, 
                  x='Mean Stability (%)',
                  y='Best Test Accuracy in Run', ratio=100)
g.plot_joint(sns.scatterplot, s=0)
g.annotate(stats.kendalltau)
kendall = g.ax_joint.get_children()[-2]
g.annotate(stats.pearsonr)
a=g.ax_joint.get_children()[-2]
a.set_zorder(5)
a.draw_frame(0)
a.set_visible(False)   
sns.regplot(data=summary,
            x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
sns.scatterplot(data=summary,
                x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), hue='Stability (Method)', palette=stability_palette,
            legend = False, s = 82)
plt.annotate(a.get_texts()[0].get_text(), (96.08, 88.34))
plt.annotate(kendall.get_texts()[0].get_text(), (96.08, 88.24))
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gcf().set_size_inches(8, 6)
g.savefig('figures_and_tables/PDFs/Figure_B2_ResNet_correlation.pdf')

# dynamics plot
data['Method'] = pd.Series([method_dict[r] for r in data.run_cfg],
                                                         index=data.index)
data['Stability (Method)']  = [stability_dict[r] for r in data['Method'] ]

plt.rcParams.update({'font.size': 16, 'legend.fontsize': 13, 
                     'legend.fancybox':True, 'legend.shadow':True})
plt.rcParams.update({'axes.labelsize':20})
g = sns.FacetGrid(data=data, hue='Stability (Method)',
                  hue_order = [stability_dict[r] for r in method_dict.values()])
g=(g.map(sns.lineplot, 'epoch', 'test'))
plt.gca().set_ylim(84,88.5)
plt.gca().set_xlim(1,325)
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gca().set_xlabel('Epoch')
plt.gcf().set_size_inches(12, 4.5)
plt.gca().tick_params(labeltop=False, labelright=True)
plt.gcf().tight_layout()
l=g.axes.flatten()[0].legend(loc='upper left',framealpha=.82, fancybox=True,
                             shadow=False,bbox_to_anchor=(.08,.96,0,0),
                             bbox_transform=plt.gcf().transFigure)
l.set_title('Stability (Method)')
plt.setp(l.get_title(), fontsize=15, weight='bold')
plt.gca().annotate('', xy=(170,85.7), xytext=(200,85.7),
            arrowprops=dict(facecolor='red', shrink=0.15), )
plt.gca().annotate('', xy=(210,85.5), xytext=(240,85.5),
            arrowprops=dict(facecolor='green', shrink=0.15), )
plt.gca().annotate('', xy=(210,85), xytext=(240,85),
            arrowprops=dict(facecolor='red', shrink=0.15), )
plt.gca().annotate('\u221227%', xy=(250,85.3), xytext=(280,85.3),va='center',
            arrowprops=dict(facecolor='green', shrink=0.05), )
plt.gca().annotate('\u221245%', xy=(250,84.8), xytext=(280,84.8),va='center',
            arrowprops=dict(facecolor='red', shrink=0.05), )
plt.gca().annotate('', xy=(280,86.2), xytext=(310,86.2),va='center',
            arrowprops=dict(facecolor='orange', shrink=0.05), )
g.axes[0][0].get_lines()[2].set_zorder(3)
plt.gcf().savefig("figures_and_tables/PDFs/Figure_B2_ResNet_dynamics.pdf",
                  bbox_inches = "tight")