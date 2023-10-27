#!/bin/sh
./train.py --epochs 200 --optimizer Adam --lr 0.001 --deterministic --compress schedule-fids.yaml --model ai85net5 --dataset FDS --device MAX78000 --print-freq 200 "$@"
