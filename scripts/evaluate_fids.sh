#!/bin/sh
./train.py --model ai85simplenet --dataset FDS --save-sample 10 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/trial/qat_checkpoint.pth.tar -8 --device MAX78000 --use-bias "$@"
