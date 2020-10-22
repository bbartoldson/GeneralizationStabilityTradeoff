#!/bin/bash

for i in {1..10}; do
        for k in {0..3}; do
		python experiments.py -r TF1_$k --arch resnet18 --adam --lr 0.0001 --epochs 4000 -d cifar100 --augment
        done
done