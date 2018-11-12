#!/usr/bin/env bash
nohup python main.py --gpu=0 --train --name=densenet-custom --densenet --noncustom &
nohup python main.py --gpu=1 --train --name=densenet-custom-beamsearch --densenet --noncustom &
