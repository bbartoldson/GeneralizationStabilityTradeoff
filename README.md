# The Generalization-Stability Tradeoff in Neural Network Pruning
This repository is the official implementation of [The Generalization-Stability Tradeoff in Neural Network Pruning](https://arxiv.org/abs/1906.03728).

![Test accuracy dynamics.](https://imgur.com/x2800gY.png)


## Requirements

Install Anaconda, then create the following environment: 

```
conda create -n GST python=3.7 scipy pandas=1.0.1 matplotlib=3.1.3 seaborn=0.10.0 pytorch=1.4 torchvision=0.5 cudatoolkit=10 -c pytorch
conda activate GST
conda install -c conda-forge pingouin
```

## Run Experiments

With the GST conda environment and from the GeneralizationStabilityTradeoff directory, execute the Bash script in the scripts_for_experiments directory that corresponds to the experiment you want to run. For example, to run the experiment associated with Figure 2, run the following command:

```
bash scripts_for_experiments/Figure_2.sh
```



## Create Figures

After your script has finished executing, the necessary data for the Figure will be stored in the logs directory, and you can create the Figure by running the corresponding graph program. For example, in the terminal inside the GeneralizationStabilityTradeoff directory, run:

```
python -m figures_and_tables.Figure_2
```

When the program is finished running, Figure 2 will appear in the figures_and_tables/PDFs directory.

As described in the Figure creation programs, some Figures require results from other experiments before they can be built. In particular, Figure 6 depends entirely on the VGG experiments that are run by scripts_for_experiments/Figure_2.sh, so if you're only interested in Figure 6, then run the Figure 2 bash script after editing out the ResNet runs. 
