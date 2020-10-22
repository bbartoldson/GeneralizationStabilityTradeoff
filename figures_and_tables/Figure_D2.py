from figures_and_tables.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", { 'grid.linestyle': '--'})


no_noise_baseline = preprocess("logs/cifar10_vgg11_bn_F2")
no_noise_baseline = no_noise_baseline.query("run_cfg=='F2_VGG_0'")
prune_L_noise = preprocess("logs/cifar10_vgg11_bn_F5")
prune_L_noise = prune_L_noise.query("run_cfg=='F5_0'")
noise = preprocess("logs/cifar10_vgg11_bn_FD2")
data = stack(noise,no_noise_baseline)
data = stack(data, prune_L_noise)

cfgs = {'F2_VGG_0': 'No Noise',
       'F5_0': 'Prune_L',
       'FD2_0': 'Zeroing 89',
       'FD2_1': 'Zeroing 630',
       'FD2_2': 'Zeroing 1220',
       'FD2_3': 'Zeroing 2480'}

# for example, cfg FD2_3 has 2480 training batches on which the weights were zeroed
print(calc_n_batches_of_training_noise(3000,40))
print(calc_n_batches_of_training_noise(1500,40))
print(calc_n_batches_of_training_noise(750,40))
print(calc_n_batches_of_training_noise(129,40))

data['Noise Detail'] = [cfgs[r] for r in data.run_cfg]
data['Noise'] = [n.split()[0] for n in data['Noise Detail']]

t = {'No Noise': sns.color_palette(n_colors = 5)[0],
       'Prune_L': sns.color_palette(n_colors = 5)[1],
       'Zeroing 89': sns.color_palette(n_colors = 6)[2],
       'Zeroing 630': sns.color_palette(n_colors = 6)[3],
      'Zeroing 1220': sns.color_palette(n_colors = 6)[4],
      'Zeroing 2480': sns.color_palette(n_colors = 6)[5]}

plt.rcParams.update({'font.size': 16, 'legend.fontsize': 13, 
                     'legend.fancybox':True, 'legend.shadow':True})
plt.rcParams.update({'axes.labelsize':20})
g = sns.FacetGrid(data=data,
                  hue='Noise Detail',
                  hue_order = list(cfgs.values()),
                  palette=t)
g=(g.map(sns.lineplot, 'epoch', 'test'))
plt.gca().set_ylim(82.5,86.25)
plt.gca().set_xlim(1,323)
plt.gca().set_ylabel('Test Accuracy (%)')
plt.gca().set_xlabel('Epoch')
plt.gcf().set_size_inches(6.5, 4.5)
plt.gcf().tight_layout()
l=g.axes.flatten()[0].legend(loc='upper left', framealpha = 0.8,
                             fancybox=True, shadow=False,bbox_to_anchor=(
                            .15,.97,0,0),bbox_transform=plt.gcf().transFigure)
l.set_title('Noise Type')
plt.setp(l.get_title(), fontsize=15, weight='bold')
plt.gcf().savefig("figures_and_tables/PDFs/Figure_D2.pdf",bbox_inches = "tight")

