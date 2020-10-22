#!/bin/bash

for i in {1..10}; do
        for k in {0..3}; do
		python experiments.py -r FD2_$k --arch vgg11_bn --adam --lr 0.001 --epochs 325 -d cifar10
        done
done
