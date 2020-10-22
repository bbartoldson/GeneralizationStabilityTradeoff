#!/bin/bash

for i in {1..10}; do
        for k in {0..5}; do
		python experiments.py -r F2_VGG_$k --arch vgg11_bn --adam --lr 0.001 --epochs 325 -d cifar10
        done
        for k in {0..3}; do
		python experiments.py -r F2_RN_$k --arch resnet18 --adam --lr 0.001 --epochs 325 -d cifar10
        done
done
