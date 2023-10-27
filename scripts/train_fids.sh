#!/bin/sh
./train.py --epochs 300 --optimizer Adam --lr 0.0008 --compress schedule-fids.yaml --model ai85simplenet --dataset FDS --device MAX78000 --batch-size 32 --print-freq 200 --use-bias "$@"
