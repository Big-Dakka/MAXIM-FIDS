#!/bin/sh
./quantize.py trained/trial/qat_best.pth.tar trained/trial/best.pth.tar --device MAX78000 -v "$@"
