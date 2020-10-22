#!/bin/bash

for i in {1..10}; do
        for k in {1..3}; do
		python experiments.py -r TC1_$k --arch resnet18 --adam --lr 0.001 --epochs 325 -d cifar10
        done
done
