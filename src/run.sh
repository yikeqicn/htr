#!/usr/bin/env bash
nohup python main.py --gpu=0 --train --name=custom &
nohup python main.py --gpu=0 --train --name=custom-bs --beamsearch &
nohup python main.py --gpu=1 --train --name=noncustom --noncustom &
nohup python main.py --gpu=1 --train --name=noncustom-bs --beamsearch --noncustom &
nohup python main.py --gpu=2 --train --name=densenet-noncustom --densenet --noncustom &
nohup python main.py --gpu=3 --train --name=densenet-noncustom-beamsearch --densenet --noncustom &

