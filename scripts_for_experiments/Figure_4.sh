#!/bin/bash

resnets=("18" "20" "56")
for i in {1..10}; do
        for r in "${resnets[@]}"; do
                for p in {0..1}; do
                        python experiments.py -r F4_${r}_${p} --arch resnet$r --epochs 165 -d cifar10 --augment
                done
        done
done        