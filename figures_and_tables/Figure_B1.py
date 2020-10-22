import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as torch_models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
tensor_and_normalize = [transforms.ToTensor(), normalize]
transform_train = transforms.Compose([ t for t in tensor_and_normalize])
trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=1)

model = torch_models.vgg19_bn(pretrained=True)
model.features = torch.nn.DataParallel(model.features) 
trained_features = model.features.module

untrained_model = torch_models.vgg19_bn(pretrained=False)
untrained_model.features = torch.nn.DataParallel(untrained_model.features) 
untrained_features = untrained_model.features.module

l = iter(trainloader)
xi, yi = next(l)
if torch.cuda.is_available():
    xi=xi.cuda()

x = np.linspace(-6,6,100)
xt = torch.tensor(x).float()
n = torch.distributions.normal.Normal(torch.tensor([0.]), torch.tensor([1.]))

def plot(ax, layer, features):
    bn_output = features[:-layer+1](xi)
    def pre_affine(channel, layer):
        x=(bn_output[:,channel] - features[-layer].bias[channel]) / features[-layer].weight[channel]
        return x.flatten().detach().cpu().numpy()
    bins = ax.hist(np.concatenate([pre_affine(channel, layer) for channel in range(512)]),
                                           bins=200,density=True)
    s = 1/bins[0].max()
    ax.plot(x, n.log_prob(xt).exp().numpy())
    
models = [untrained_features, trained_features]
plt.gcf().tight_layout()
f,a = plt.subplots(8,2)
for idx, row in enumerate(a):
    for model_id, ax in enumerate(row):
        if idx>=4:
            num = 4+idx*3
        else:
            num = 3+idx*3
        plot(ax, num, models[model_id])
        ax.set_xlim([-5,5])
        ax.set_ylim([0,.8])
        ax.set_aspect(10)
fig = plt.gcf()
fig.set_size_inches(6,18)
for ax in fig.axes:
    ax.set_xlabel('Activation Magnitude')
    ax.set_ylabel('PDF')
fig.text(0.0, 0.5, 'VGG19 Layer Depth', ha='center', va='center', rotation='vertical')
fig.text(0.3, 0.00, 'Untrained VGG19', ha='center', va='center')
fig.text(0.75, 0.00, 'Trained VGG19', ha='center', va='center')
fig.tight_layout()
fig.savefig('figures_and_tables/PDFs/Figure_B1.pdf',bbox_inches = "tight")
