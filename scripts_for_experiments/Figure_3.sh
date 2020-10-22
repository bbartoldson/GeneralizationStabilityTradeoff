#!/bin/bash

retrains=("_4" "_20" "_40" "_60" "_100" "_200" "_300")
targets=("_S" "_R" "_L")
for i in {1..5}; do
        for k in "${targets[@]}"; do
                for p in "${retrains[@]}"; do
                        python experiments.py -r F3$k$p --epochs 325 --arch vgg11_bn --adam --lr 0.001
                done
        done
done
~           
