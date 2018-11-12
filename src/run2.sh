#!/usr/bin/env bash
nohup python main.py --gpu=0 --train --name=densenet-noncustom-beamsearch --densenet --beamsearch --reduction=.4 &
nohup python main.py --gpu=0 --train --name=densenet-noncustom-beamsearch --densenet --beamsearch --depth=56 &
nohup python main.py --gpu=1 --train --name=densenet-noncustom-beamsearch --densenet --beamsearch --growth_rate=9 &
nohup python main.py --gpu=1 --train --name=densenet-noncustom-beamsearch --densenet --beamsearch --growth_rate=7 &
