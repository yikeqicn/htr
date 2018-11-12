#!/usr/bin/env bash
nohup python main.py --gpu=0 --train --name=reduction2 --densenet --beamsearch --reduction=.5 &
nohup python main.py --gpu=0 --train --name=depth2 --densenet --beamsearch --depth=66 &
nohup python main.py --gpu=1 --train --name=growthrate-14 --densenet --beamsearch --growth_rate=14 &
nohup python main.py --gpu=1 --train --name=growthrate-5 --densenet --beamsearch --growth_rate=5 &
nohup python main.py --gpu=2 --train --name=reduction3 --densenet --beamsearch --reduction=.4 &
nohup python main.py --gpu=2 --train --name=depth3 --densenet --beamsearch --depth=76 &
nohup python main.py --gpu=3 --train --name=growthrate-16 --densenet --beamsearch --growth_rate=16 &
nohup python main.py --gpu=3 --train --name=growthrate-3 --densenet --beamsearch --growth_rate=3 &

