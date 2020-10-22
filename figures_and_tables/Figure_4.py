from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


for resnum in ['18', '20', '56']:
    data = preprocess("logs/cifar10_resnet" + resnum + "_F4")
    data = add_pruning_status_vars(data)
    data = add_stability_vars(data)
    summary = summarize(data)
    
    
    styles = {'F4_'+resnum+'_0':'Prune_S 3%',
              'F4_'+resnum+'_1':'Prune_S 5%' }
    summary['Pruning Detail'] = [styles[r] for r in summary.run_cfg]
    stable_dict = dict([(k, (str(round(v,2)) if v!=100 else '100') + '% Stable (' + k + ')')
          for k, v in summary.groupby("Pruning Detail")['Mean Stability (%)'].mean().items()])
    summary['Stability (Method)'] = [stable_dict[k] for k in summary['Pruning Detail']]
    
    
    r_palette ={'F4_'+resnum+'_0':sns.color_palette(n_colors = 6)[0],
                'F4_'+resnum+'_1':sns.color_palette(n_colors = 6)[1]
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
    if resnum == '18':
        plt.annotate(a.get_texts()[0].get_text(), (94.2,94.11))
        plt.annotate(kendall.get_texts()[0].get_text(), (94.2,94.05))
    elif resnum == '20':
        plt.annotate(a.get_texts()[0].get_text(), (71, 90.93))
        plt.annotate(kendall.get_texts()[0].get_text(), (71,90.84))
    elif resnum == '56':
        plt.annotate(a.get_texts()[0].get_text(), (65, 91.6))
        plt.annotate(kendall.get_texts()[0].get_text(), (65,91.3))
    g.savefig('figures_and_tables/PDFs/Figure_4_Gen_ResNet'+resnum+'.pdf')
    
    
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
    if resnum == '18':  
        plt.annotate(a.get_texts()[0].get_text(), (94.2, 5.92))
        plt.annotate(kendall.get_texts()[0].get_text(), (94.2, 5.85))
    elif resnum == '20':
        plt.annotate(a.get_texts()[0].get_text(), (76, 7.08))
        plt.annotate(kendall.get_texts()[0].get_text(), (76,7.015))
    elif resnum == '56':
        plt.annotate(a.get_texts()[0].get_text(), (65, 8.35))
        plt.annotate(kendall.get_texts()[0].get_text(), (65,8.13))
    g.savefig('figures_and_tables/PDFs/Figure_4_Gap_ResNet'+resnum+'.pdf')