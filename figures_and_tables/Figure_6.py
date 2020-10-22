from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


# preprocess data 
data = preprocess("logs/cifar10_vgg11_bn_F2")
data = add_stability_vars(data)
data = add_pruning_status_vars(data)
summary = summarize(data)
flatness = data.query("epoch==315")[['run_cfg','run','grad_covariance_trace',
                                                     'hessian_spectrum']]
summary = pd.merge(summary, flatness)

detail = {'F2_VGG_0': 'No Pruning',
       'F2_VGG_1': 'Prune_S 1%',
       'F2_VGG_2': 'Prune_S 13%',
       'F2_VGG_3': 'Prune_R 13%',
       'F2_VGG_4': 'Prune_L 13%',
       'F2_VGG_5': 'Scratch Pruning'}
summary['Pruning Detail'] = [detail[k] for k in summary.run_cfg]
t = {'No Pruning': sns.color_palette(n_colors = 6)[0],
       'Prune_S 1%': sns.color_palette(n_colors = 6)[1],
        'Prune_S 13%': sns.color_palette(n_colors = 6)[2],
        'Prune_R 13%': sns.color_palette(n_colors = 6)[4],
        'Prune_L 13%': sns.color_palette(n_colors = 6)[3],
       'Scratch Pruning': sns.color_palette(n_colors = 6)[5]}

# truncating the eigenspectrum is giving effectively the same result as just
# using the first 100 eigenvalues
summary['hessian_first_100_eigen_sum'] = [ sum(s) for s in summary['hessian_spectrum']]
summary['hessian_truncated_eigenspectrum_sum'] = [ sum(s[abs(s)>6e-3*max(abs(s))])
                                               for s in summary['hessian_spectrum']]
g=sns.jointplot(x='hessian_first_100_eigen_sum', y='hessian_truncated_eigenspectrum_sum',
                data=summary,kind="reg")
g.annotate(stats.pearsonr)
sum(abs((summary['hessian_first_100_eigen_sum']-summary['hessian_truncated_eigenspectrum_sum'])
    /summary['hessian_first_100_eigen_sum']) > 0.005)
sum(abs((summary['hessian_first_100_eigen_sum']-summary['hessian_truncated_eigenspectrum_sum'])
    /summary['hessian_first_100_eigen_sum']) > 0.01)
  

# hessian plots
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 16})
plt.rcParams.update({'axes.labelsize':20})
g = sns.JointGrid(data=summary,
                  x='Mean Stability (%)',
                  y='hessian_first_100_eigen_sum', ratio=100)
g.plot_joint(sns.scatterplot, s=0)
g.annotate(stats.kendalltau)
kendall = g.ax_joint.get_children()[-2]
g.annotate(stats.pearsonr)
a=g.ax_joint.get_children()[-2]
a.set_zorder(5)
a.draw_frame(0)
a.set_visible(False)
plt.gca().set_autoscale_on(False)
sns.regplot(data=summary,
            x='Mean Stability (%)', y='hessian_first_100_eigen_sum',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
sns.scatterplot(data=summary,
                x='Mean Stability (%)', y='hessian_first_100_eigen_sum',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette=t,
            legend =False, s=100)
plt.gca().set_ylabel( r'Curvature,' '\n' r'Sensitivity of $\nabla L_w$ to $w$')
plt.gca().annotate(r'$Flatter$',xy=(98.27,205), color='g',
            annotation_clip=False)
plt.gca().annotate(r'$Sharper$',xy=(98.25,538), color='r',
            annotation_clip=False)
plt.gca().set_ylim(205,555)
plt.gca().set_yticks(range(250,550,50))
plt.gcf().set_size_inches(8, 6)
plt.annotate(a.get_texts()[0].get_text(), (98.8, 515))
plt.annotate(kendall.get_texts()[0].get_text(), (98.8, 465))
g.savefig('figures_and_tables/PDFs/Figure_6_Stability_Hessian.pdf')


plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':22})
g = sns.JointGrid(data=summary,
                  x='hessian_first_100_eigen_sum',
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
sns.regplot(data=summary,
            x='hessian_first_100_eigen_sum', y='Best Test Accuracy in Run',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
s = sns.scatterplot(data=summary,
                x='hessian_first_100_eigen_sum', y='Best Test Accuracy in Run',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t,
            legend=False,
            s=100)
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gca().set_xlabel( r'Curvature,' '\n' r' Sensitivity of $\nabla L_w$ to $w$')
plt.gcf().set_size_inches(8, 6)
plt.gca().annotate(r'$Flatter$',xy=(205,84.54), color='g', 
            annotation_clip=False)
plt.gca().annotate(r'$Sharper$',xy=(490,84.54), color='r',
            annotation_clip=False)
plt.gca().set_xlim(205,555)
plt.gca().set_xticks(range(250,550,50))
plt.gcf().set_size_inches(8, 6)
plt.annotate(a.get_texts()[0].get_text(), ( 310,86.27))
plt.annotate(kendall.get_texts()[0].get_text(), (301, 86.05))
g.savefig('figures_and_tables/PDFs/Figure_6_Acc_Hessian.pdf')


# make legend
stable_dict = dict([(k, (str(round(v,2)) if v!=100 else '100') + '% Stable (' + k + ')')
      for k, v in summary.groupby("Pruning Detail")['Mean Stability (%)'].mean().items()])
summary['Stability (Method)'] = [stable_dict[k] for k in summary['Pruning Detail']]
stability_palette = dict([(v, t[k]) for k, v in stable_dict.items()])

s = sns.scatterplot(data=summary,
                x='hessian_first_100_eigen_sum', y='Best Test Accuracy in Run',
            hue='Stability (Method)',
            hue_order = [stable_dict[r] for r in detail.values()],
            palette = stability_palette,
            s=100)
legend = plt.legend(s.legend_.legendHandles[1:],
                    [stable_dict[r] for r in detail.values()],
           loc='lower center', bbox_to_anchor=(-0.5, -0.5), fontsize=18,
           shadow=False, ncol = 6, frameon = False)
for handle in legend.legendHandles:
    handle.set_sizes([85.0])
fig=legend.figure
fig.canvas.draw()
bbox  = s.legend_.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("figures_and_tables/PDFs/Figure_6_Legend.pdf", 
            dpi="figure", bbox_inches=bbox)


# grad covariance plots
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':19})
g = sns.JointGrid(data=summary,
                  x='Mean Stability (%)',
                  y='grad_covariance_trace', ratio=100)
g.plot_joint(sns.scatterplot, s=0)
g.annotate(stats.kendalltau)
kendall = g.ax_joint.get_children()[-2]
g.annotate(stats.pearsonr)
a=g.ax_joint.get_children()[-2]
a.set_zorder(5)
a.draw_frame(0)
a.set_visible(False)
plt.gca().set_autoscale_on(False)
sns.regplot(data=summary,
            x='Mean Stability (%)', y='grad_covariance_trace',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
sns.scatterplot(data=summary,
                x='Mean Stability (%)', y='grad_covariance_trace',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette=t,
            legend =False, s=100)
plt.gca().set_ylabel( r'Noise,' '\n' r'Sensitivity of $\nabla L_w$ to Sample')
plt.gcf().set_size_inches(8, 6)
plt.gca().annotate(r'$Flatter$',xy=(98.28,1320), color='g', 
            annotation_clip=False)
plt.gca().annotate(r'$Sharper$',xy=(98.26,5300), color='r',
            annotation_clip=False)
plt.annotate(a.get_texts()[0].get_text(), ( 98.79,4500))
plt.annotate(kendall.get_texts()[0].get_text(), (98.79, 3400))
g.savefig('figures_and_tables/PDFs/Figure_6_Stability_Gradient.pdf')


plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':22})
g = sns.JointGrid(data=summary,
                  x='grad_covariance_trace',
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
sns.regplot(data=summary,
            x='grad_covariance_trace', y='Best Test Accuracy in Run',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
s = sns.scatterplot(data=summary,
                x='grad_covariance_trace', y='Best Test Accuracy in Run',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t,
            legend=False,
            s=100)
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gca().set_xlabel( r'Noise,' '\n' r' Sensitivity of $\nabla L_w$ to Sample')
plt.gcf().set_size_inches(8, 6)
plt.gca().annotate(r'$Flatter$',xy=(1250,84.57), color='g', 
            annotation_clip=False)
plt.gca().annotate(r'$Sharper$',xy=(4750,84.57), color='r',
            annotation_clip=False)
plt.annotate(a.get_texts()[0].get_text(), ( 2450,86.35))
plt.annotate(kendall.get_texts()[0].get_text(), (2450, 86.1))
g.savefig('figures_and_tables/PDFs/Figure_6_Acc_Gradient.pdf')