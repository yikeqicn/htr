#!/usr/bin/env bash
nohup python main.py --gpu=0 --train --name=reduction --densenet --beamsearch --reduction=.4 &
nohup python main.py --gpu=0 --train --name=depth --densenet --beamsearch --depth=56 &
nohup python main.py --gpu=1 --train --name=growthrate-9 --densenet --beamsearch --growth_rate=9 &
nohup python main.py --gpu=1 --train --name=growthrate-7 --densenet --beamsearch --growth_rate=7 &

nohup python main.py --gpu=2 --train --name=dropout --densenet --beamsearch --keep_prob=.5 &
