from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


####################
# preprocess data #
###################

data = preprocess("logs/cifar10_vgg11_bn_F2")
data = add_stability_vars(data)
data = add_pruning_status_vars(data)
summary = summarize(data)
flatness = data.query("epoch==315")[['run_cfg','run','grad_covariance_trace','hessian_spectrum']]
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


# one observation per eigenvalue
summary['magnitude_sorted_hessian_spectrum'] = [sorted(h, key=lambda x: -abs(x)) for
                                                 h in summary.hessian_spectrum]
exploded = summary.explode("magnitude_sorted_hessian_spectrum").reset_index()
exploded.rename( columns={'magnitude_sorted_hessian_spectrum' :'Eigenvalue'}, inplace=True )
exploded['Eigenvalue'] = exploded['Eigenvalue'].astype(float)
exploded['Eigenvalue Magnitude'] = abs(exploded['Eigenvalue'])
exploded['Eigenvalue Magnitude Index'] = (exploded.index.values%100).astype(float)


plt.rcParams.update({'font.size': 16, 'legend.fontsize': 13, 
                     'legend.fancybox':True, 'legend.shadow':True})
plt.rcParams.update({'axes.labelsize':20})
sns.relplot(x="Eigenvalue Magnitude Index",y="Eigenvalue",
            kind='line',
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t,
            ci = 80,
            data=exploded)
plt.xscale("symlog")
plt.title("Hessian Spectrum at Convergence")
plt.savefig('figures_and_tables/PDFs/Figure_E2_Eigenvalue.pdf',
            bbox_inches = "tight")

sns.relplot(x="Eigenvalue Magnitude Index",y="Eigenvalue Magnitude",
            kind='line',
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t,
            ci = None,
            data=exploded)
plt.title("Hessian Spectrum at Convergence")
plt.savefig('figures_and_tables/PDFs/Figure_E2_Eigenvalue_Magnitude.pdf',
            bbox_inches = "tight")


plt.rcParams.update({'font.size': 16, 'legend.fontsize': 13, 'legend.frameon':True,
                     'legend.fancybox':False, 'legend.shadow':False})
plt.rcParams.update({'axes.labelsize':20})
sns.catplot(x="Eigenvalue Magnitude Index",y="Eigenvalue",
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t, aspect =50, kind='bar',
            data=exploded[exploded["Eigenvalue Magnitude Index"]<10])
plt.gcf().set_size_inches(8, 6)
plt.xticks(ticks=range(10),labels=range(10))
plt.savefig('figures_and_tables/PDFs/Figure_E1.pdf')


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
plt.gca().set_autoscale_on(False)
sns.regplot(data=summary,
            x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
sns.scatterplot(data=summary,
                x='Mean Stability (%)', y='Best Test Accuracy in Run',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t,
            legend =False, s=100)
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gcf().set_size_inches(8, 6)
plt.annotate(a.get_texts()[0].get_text(), (98.8, 85.37))
plt.annotate(kendall.get_texts()[0].get_text(), (98.8, 85.17))
g.savefig('figures_and_tables/PDFs/Figure_E6.pdf')






summary['TIC'] = summary.grad_covariance_trace/summary.hessian_first_100_eigen_sum
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':19})
g = sns.JointGrid(data=summary,
                  x='Mean Stability (%)',
                  y='TIC', ratio=100)
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
            x='Mean Stability (%)', y='TIC',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
sns.scatterplot(data=summary,
                x='Mean Stability (%)', y='TIC',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette=t,
            legend =False, s=100)
plt.annotate(a.get_texts()[0].get_text(), (98.9,11.5))
plt.annotate(kendall.get_texts()[0].get_text(), (98.9,10.5))
plt.gca().set_ylabel( r'Tr($\mathbf{C}$) / Tr($\mathbf{H}_{100}$)')
plt.gcf().set_size_inches(8, 6)
g.savefig('figures_and_tables/PDFs/Figure_E5_Stability_TIC.pdf')


plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':22})
g = sns.JointGrid(data=summary,
                  x='TIC',
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
            x='TIC', y='Best Test Accuracy in Run',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
s = sns.scatterplot(data=summary,
                x='TIC', y='Best Test Accuracy in Run',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t,
            legend=False,
            s=100)
plt.annotate(a.get_texts()[0].get_text(), (6.8, 86.5))
plt.annotate(kendall.get_texts()[0].get_text(), (6.7, 86.02))
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gca().set_xlabel( r'Tr($\mathbf{C}$) / Tr($\mathbf{H}_{100}$)')
plt.gcf().set_size_inches(8, 6)
g.savefig('figures_and_tables/PDFs/Figure_E5_Acc_TIC.pdf')



# epsilon graphs
def build_perturb_dict(df):
    new = {'run_cfg':[], 'run':[],  'eigenvector_index':[],
           'perturbation':[],
           'loss': [], 'new_loss':[], 'loss_increase':[]}
    df = df.query("epoch==315")
    (post_perturb_test_acc, post_perturb_test_loss, 
     test_acc,test_loss) = (df.post_perturb_acc.values, 
                                                         df.post_perturb_loss.values,
                             df.test.values, df.test_loss.values)
    for r in df.run_cfg.unique():
        run_cfg_runs = np.nonzero(r==df.run_cfg.values)[0]
        for run, dictio in zip(df.run.values[run_cfg_runs], 
                                     post_perturb_test_acc[run_cfg_runs]):
            for k, v in dictio.items():
                new['run_cfg'].append(r)
                new['run'].append(run)
                new['eigenvector_index'].append(k[0])
                new['perturbation'].append(k[1])
        for run, (dictio, test) in enumerate(zip(post_perturb_test_loss[run_cfg_runs],
                                         test_loss[run_cfg_runs])):
            for k, v in dictio.items():
                new['new_loss'].append(v)
                new['loss'].append(test)    
                new['loss_increase'].append(v-test)
                
    return new
    

perturbation_effects = pd.DataFrame(build_perturb_dict(data))
perturbation_effects['Pruning Detail'] = [detail[k] for k in perturbation_effects.run_cfg]

plt.rcParams.update({'axes.labelsize':13})
plt.rcParams.update({'font.size': 13})
g = sns.FacetGrid(data=perturbation_effects.query("abs(perturbation)<.13").query("eigenvector_index!=9"),
            hue = "Pruning Detail", palette=t,
                  col = "eigenvector_index", col_wrap=3)
g=(g.map(sns.lineplot, 'perturbation', 'loss_increase', 
         ci=None))
def translate(s):
    tex = "$\ \ (v_" + str(int(s[-1])) + ")$"
    return 'Eigenvector '+s[-1] +" "+ tex
g.fig.tight_layout()
for ax in g.axes:
    ax.set_ylabel("Increase in Test Loss")
    ax.set_xlabel(r"$\epsilon\quad (w = w^\ast + \epsilon v_i)$")
    ax.set_title(translate(ax.get_title()))
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.gcf().savefig("figures_and_tables/PDFs/Figure_E4.pdf",bbox_inches = "tight")



# use the following data to get epsilon you can perturb before a loss increase of 0.1 occurs
perturbation_effects['abs_perturbation'] = abs(perturbation_effects.perturbation)
perturbation_effects_w_mean_stability = pd.merge(perturbation_effects,summary)
dic={}
for cfg in  perturbation_effects_w_mean_stability.run_cfg.unique():
    dic[cfg] = perturbation_effects_w_mean_stability[
                perturbation_effects_w_mean_stability.run_cfg==cfg]['run'].unique()

worst = perturbation_effects_w_mean_stability.groupby(
    ['run','run_cfg','abs_perturbation','Mean Stability (%)','Pruning Detail',
     'Best Test Accuracy in Run'])['loss_increase'].max().reset_index()
worst = worst[worst.loss_increase>=0.1]
epsilons = worst.groupby(['run','run_cfg','Mean Stability (%)','Pruning Detail',
            'Best Test Accuracy in Run'])['abs_perturbation'].min().reset_index()
assert len(epsilons == len(perturbation_effects.drop_duplicates(['run','run_cfg']))), print(
    'confirming that every run had a perturbation large enough to change loss by 0.1')



plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':19})
g = sns.JointGrid(data=epsilons,
                  x='Mean Stability (%)',
                  y='abs_perturbation', ratio=100)
g.plot_joint(sns.scatterplot, s=0)
g.annotate(stats.kendalltau)
kendall = g.ax_joint.get_children()[-2]
g.annotate(stats.pearsonr)
a=g.ax_joint.get_children()[-2]
a.set_zorder(5)
a.draw_frame(0)
a.set_visible(False)
plt.gca().set_autoscale_on(False)
sns.regplot(data=epsilons,
            x='Mean Stability (%)', y='abs_perturbation',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
sns.scatterplot(data=epsilons,
                x='Mean Stability (%)', y='abs_perturbation',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette=t,
            legend =False, s=100)
plt.annotate(a.get_texts()[0].get_text(), (98.9, .145))
plt.annotate(kendall.get_texts()[0].get_text(), (98.9, .135))
plt.gca().set_ylabel(r'$\delta w$ at Loss Increase of 0.1')
plt.gcf().set_size_inches(8, 6)
g.savefig('figures_and_tables/PDFs/Figure_E3_Stability_Epsilon.pdf')


plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
plt.rcParams.update({'axes.labelsize':22})
g = sns.JointGrid(data=epsilons,
                  x='abs_perturbation',
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
sns.regplot(data=epsilons,
            x='abs_perturbation', y='Best Test Accuracy in Run',
            ax = plt.gca(), color="grey", ci=None,scatter=False)
s = sns.scatterplot(data=epsilons,
                x='abs_perturbation', y='Best Test Accuracy in Run',
            ax = plt.gca(), 
            hue='Pruning Detail',
            hue_order = list(detail.values()),
            palette = t,
            legend=False,
            s=100)
plt.annotate(a.get_texts()[0].get_text(), (.095, 86.4))
plt.annotate(kendall.get_texts()[0].get_text(), (.095, 86.27))
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gca().set_xlabel(r'$\delta w$ at Loss Increase of 0.1')
plt.gcf().set_size_inches(8, 6)
g.savefig('figures_and_tables/PDFs/Figure_E3_Acc_Epsilon.pdf')