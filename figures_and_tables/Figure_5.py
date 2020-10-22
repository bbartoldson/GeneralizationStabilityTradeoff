from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", { 'grid.linestyle': '--'})

# bring in the no noise, no pruning baseline from the Figure 2 data. 
# you can just run the F2_VGG_0 cfg, you don't need to recreate all the
# Figure 2 data to get this file. 
no_noise_baseline = preprocess("logs/cifar10_vgg11_bn_F2")
no_noise_baseline = no_noise_baseline.query("run_cfg=='F2_VGG_0'")
noise = preprocess("logs/cifar10_vgg11_bn_F5")
data = stack(noise,no_noise_baseline)

cfgs = {'F2_VGG_0': 'No Noise',
       'F5_0': 'Prune_L',
       'F5_4': 'Gaussian 1',
       'F5_5': 'Gaussian 50',
       'F5_6': 'Gaussian 1105',
       'F5_1': 'Zeroing 1',
       'F5_2': 'Zeroing 50',
       'F5_3': 'Zeroing 1105'}

# for example, cfg F5_2 has 50 training batches on which the weights were zeroed
print(calc_n_batches_of_training_noise(1500,79))
print(calc_n_batches_of_training_noise(129,79))


data['Noise Detail'] = [cfgs[r] for r in data.run_cfg]
data['Noise'] = [n.split()[0] for n in data['Noise Detail']]

t = {'No Noise': sns.color_palette(n_colors = 5)[0],
       'Prune_L': sns.color_palette(n_colors = 5)[1],
        'Gaussian 1': sns.color_palette(n_colors = 5)[2],
        'Gaussian 50': sns.color_palette(n_colors = 5)[3],
        'Gaussian 1105': sns.color_palette(n_colors = 5)[4],
       'Zeroing 1': sns.color_palette(n_colors = 5)[2],
       'Zeroing 50': sns.color_palette(n_colors = 5)[3],
      'Zeroing 1105': sns.color_palette(n_colors = 5)[4]}

plt.rcParams.update({'font.size': 16, 'legend.fontsize': 13, 'legend.fancybox':True,
                         'legend.shadow':True})
plt.rcParams.update({'axes.labelsize':20})
def plot(noise):
    drop = 'Gaussian'
    if noise == 'Gaussian':
        drop = 'Zeroing'
    temp = data[data.Noise!=drop]
    # we set pruned weights to zero on the last epoch, which harms acc of noise methods
    temp = temp.query("epoch<325")
    g = sns.FacetGrid(data=temp,
                      hue='Noise Detail',
                      hue_order = list(cfgs.values()),
                      palette=t)
    g=(g.map(sns.lineplot, 'epoch', 'test'))
    plt.gca().set_ylim(82.5,86.25)
    plt.gca().set_xlim(1,324)
    plt.gca().set_ylabel('Test Accuracy (%)')
    plt.gca().set_xlabel('Epoch')
    plt.gcf().set_size_inches(6.5, 4.5)
    plt.gcf().tight_layout()
    l=g.axes.flatten()[0].legend(loc='upper left', fancybox=True,
        shadow=True,bbox_to_anchor=(.15,.97,0,0),bbox_transform=plt.gcf().transFigure)
    l.set_title('Noise Type')
    plt.setp(l.get_title(), fontsize=15, weight='bold')
    if noise == 'Zeroing':
        g.axes[0][0].get_lines()[1].set_zorder(4)
        g.axes[0][0].get_lines()[3].set_zorder(4)
    plt.gcf().savefig("figures_and_tables/PDFs/Figure_5_" + noise + ".pdf",
                      bbox_inches = "tight")
plot('Zeroing')
plot('Gaussian')
