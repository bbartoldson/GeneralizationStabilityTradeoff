from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pingouin as pg
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


####################
# preprocess data #
###################

data = preprocess('logs/cifar10_vgg11_bn_F3')
data['Retrain'] = pd.Series([int(r.split('_')[-1]) for r in data.run_cfg], 
                            index=data.index)
data['Iterative_Pruned_Percent'] = pd.Series([90/len(range(6,275,r)) for r
                                              in data.Retrain], index=data.index)
data['Pruning Style'] = ['Prune_' + r.split('_')[1] for r in data.run_cfg]
data = add_pruning_status_vars(data)
data = add_stability_vars(data)
summary = summarize(data, ['Retrain', 'Pruning Style', 'Iterative_Pruned_Percent'])
summary['Iterative Pruning Rate'] = round(summary.Iterative_Pruned_Percent,1)


#############
# plotting #
############

plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
plt.rcParams.update({'axes.labelsize':20})
g = sns.FacetGrid(data=data.query('post_prune').groupby(
        ['Retrain', 'Pruning Style', 'Iterative_Pruned_Percent','run']
        )['test'].max().reset_index(), 
                  hue_order=['Prune_S', 'Prune_R', 'Prune_L'],
    legend_out=True, size=5.8, aspect=1.25)
g=(g.map_dataframe(sns.lineplot, 'Iterative_Pruned_Percent', 'test',
                   hue = 'Pruning Style', 
                   markers=True,style='Pruning Style', 
                  hue_order=['Prune_S', 'Prune_R', 'Prune_L'],
                   ci=95).add_legend())
g.fig.get_children()[-1].set_bbox_to_anchor((.86, 0.28, 0, 0))
g.fig.tight_layout()
g.facet_axis(0,0).set_ylim(85.3,86.38)
g.axes[0][0].set_xscale('symlog',basex=2)
g.axes[0][0].set_xticklabels([2**x for x in range(0,8)])
for row in g.axes:
    for ax in row:
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_xlabel('Iterative Pruning Rate (% and Log Scale)')
g.savefig("figures_and_tables/PDFs/Figure_3_Accuracy.pdf")



g = sns.FacetGrid(data=data.query("pruned_epoch==True").groupby(
        ['Retrain', 'Pruning Style', 'Iterative_Pruned_Percent','run']
        ).mean().Stability.reset_index(),
    legend_out=False, size=5.8, aspect=1.25)
g=(g.map_dataframe(sns.lineplot, 'Iterative_Pruned_Percent', 
                   'Stability', hue = 'Pruning Style',
                  hue_order=['Prune_S', 'Prune_R', 'Prune_L'],
                  markers=True,style='Pruning Style'))
g.fig.tight_layout()
g.axes[0][0].set_xscale('symlog',basex=2)
for row in g.axes:
    for ax in row:
        ax.set_ylabel('Mean Stability (%)')
        ax.set_xlabel('Iterative Pruning Rate (% and Log Scale)')
g.axes[0][0].set_xticklabels([2**x for x in range(0,8)])
ax_ins = inset_axes(g.axes[0][0],
                    width="54.3%", 
                    height="55%", 
                    loc="center",bbox_to_anchor=(-.037, -.05, 1, 1),
                      bbox_transform=g.axes[0][0].transAxes)
ax_ins.set_ylim(98.65,100)
sns.lineplot('Iterative_Pruned_Percent',
                   'Stability', hue = 'Pruning Style', legend=False,
                  hue_order=['Prune_S', 'Prune_R', 'Prune_L'], markers=True,
                  style='Pruning Style',
                  data=data.query("Retrain<72").query("pruned_epoch==True").groupby(
        ['Retrain', 'Pruning Style', 'Iterative_Pruned_Percent','run']
        ).mean().Stability.reset_index(), ax=ax_ins)
ax_ins.set_ylabel("")
ax_ins.set_xlabel("")
ax_ins.set_xscale('symlog',basex=2)
ax_ins.set_xticks([2**x for x in range(1,5)])
ax_ins.set_xticklabels([2**x for x in range(1,5)])
g.savefig("figures_and_tables/PDFs/Figure_3_Stability.pdf")



# we will graph the pearson and kendall correlation coefficient information
corrs = []
k_corrs = []
cis = []
k_cis = []

# compute the correlation at each retrain period (and thereby iterative rate)
# also compute the confidence interval for each correlation
for r in [4, 20, 40, 60, 100, 200, 300]:
    g = sns.JointGrid(data=summary[summary.Retrain==r],
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
    plt.gca().set_autoscale_on(False)
    sns.regplot(data=summary[summary.Retrain==r],x='Mean Stability (%)', 
                y='Best Test Accuracy in Run',
                ax = plt.gca(), color="grey", ci=None,scatter=False,
                line_kws={'linewidth':6})
    if r <20:
        sns.scatterplot(data=summary[summary.Retrain==r],x='Mean Stability (%)', 
                        y='Best Test Accuracy in Run',
                    ax = plt.gca(), hue='Pruning Style',
                          hue_order=['Prune_S', 'Prune_R', 'Prune_L'],
                    legend ='brief', s=82)
    else:
        sns.scatterplot(data=summary[summary.Retrain==r],x='Mean Stability (%)',
                        y='Best Test Accuracy in Run',
                    ax = plt.gca(), hue='Pruning Style',
                          hue_order=['Prune_S', 'Prune_R', 'Prune_L'],
                    legend =False, s=82)
    plt.gca().set_ylabel('Test Accuracy (%)')
    plt.gca().set_xlabel('Mean Stability (%)')
    plt.gcf().set_size_inches(8, 6)
    g.ax_joint.legend_.set_bbox_to_anchor((0.515,.65,0,0))
    if r ==4:
        plt.annotate(a.get_texts()[0].get_text(), (99.7, 85.35))
        plt.annotate(kendall.get_texts()[0].get_text(), (99.7, 85.21))
    elif r ==20:
        plt.annotate(a.get_texts()[0].get_text(), (98.50, 85.5))
        plt.annotate(kendall.get_texts()[0].get_text(), (98.50, 85.3))
    elif r ==40:
        plt.annotate(a.get_texts()[0].get_text(), (98., 85.7))
        plt.annotate(kendall.get_texts()[0].get_text(), (98., 85.5))
    elif r ==60:
        plt.annotate(a.get_texts()[0].get_text(), (98., 85.5))
        plt.annotate(kendall.get_texts()[0].get_text(), (98., 85.3))
    elif r ==100:
        plt.annotate(a.get_texts()[0].get_text(), (97.3, 85.7))
        plt.annotate(kendall.get_texts()[0].get_text(), (97.3, 85.61))
    elif r ==200:
        plt.annotate(a.get_texts()[0].get_text(), (91, 85.62))
        plt.annotate(kendall.get_texts()[0].get_text(), (91, 85.55))
    elif r ==300:
        plt.annotate(a.get_texts()[0].get_text(), (81, 85.14))
        plt.annotate(kendall.get_texts()[0].get_text(), (81, 85.05))
    corrs.append(a.get_texts()[0].get_text())
    k_corrs.append(kendall.get_texts()[0].get_text())
    k_cis.append(pg.corr(summary[summary.Retrain==r]['Mean Stability (%)'],
                   summary[summary.Retrain==r]['Best Test Accuracy in Run'],
                   method='kendall').values[0][2])
    cis.append(pg.corr(summary[summary.Retrain==r]['Mean Stability (%)'],
                   summary[summary.Retrain==r]['Best Test Accuracy in Run']
                   ).values[0][2])
    g.savefig("figures_and_tables/PDFs/Figure_C1_Accuracy_Stability_Retrain_"
              + str(r) + ".pdf")
        
for corr_type, corr_list, ci_list in zip(['Kendall', 'Pearson'],
                                         [k_corrs, corrs],
                                         [k_cis, cis]):
    # set up a dictionary with the correlation at each rate
    di = {}
    di['rate'] = np.sort(summary['Iterative Pruning Rate'].unique())
    di['corr'] = [float(c.split()[2][:-1]) for c in corr_list]
    corr_summary = pd.DataFrame(di)
    
    # x locations for iterative rates
    x =  list(di['rate'][1:-1]) 
    x.insert(0,(di['rate']-0.05)[0])
    x.insert(len(x),(di['rate']+0.97)[-1])
    
    # graph the correlations with confidence intervals
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
    plt.rcParams.update({'axes.labelsize':20})
    g = sns.FacetGrid(data=corr_summary,
                      size=5.8, aspect=1.25)
    g=(g.map_dataframe(sns.lineplot, 'rate', 'corr', linewidth=4).add_legend())
    g.fig.get_children()[-1].set_bbox_to_anchor((.86, 0.28, 0, 0))
    g.fig.tight_layout()
    g.axes[0][0].set_xscale('symlog',basex=2)
    g.axes[0][0].set_xticklabels([2**x for x in range(0,8)])
    lower_bound = [a[0] for a in ci_list]
    upper_bound = [a[1] for a in ci_list]
    plt.fill_between(x, lower_bound, upper_bound, alpha=.21)
    for row in g.axes:
        for ax in row:
            ax.set_ylabel("Correlation between\nGeneralization and Stability")
            ax.set_xlabel('Iterative Pruning Rate (% and Log Scale)')
    if corr_type!='Kendall':
        g.savefig("figures_and_tables/PDFs/Figure_3_Corr_"+corr_type+".pdf")
    else:  
        g.savefig("figures_and_tables/PDFs/Figure_C2.pdf")