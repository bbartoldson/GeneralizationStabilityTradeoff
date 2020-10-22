#!/bin/bash

for i in {1..10}; do
            for p in {0..2}; do
                    python experiments.py -r FC3_${p} --arch resnet18 --epochs 165 -d cifar10 --augment
            done
done        