#!/usr/bin/env bash
python main.py --gpu=0 --train --name=custom
python main.py --gpu=0 --train --name=custom-bs --beamsearch
python main.py --gpu=1 --train --name=noncustom --noncustom
python main.py --gpu=1 --train --name=noncustom-bs --beamsearch --noncustom
python main.py --gpu=2 --train --name=densenet-noncustom --densenet --noncustom
python main.py --gpu=3 --train --name=densenet-noncustom-beamsearch --densenet --noncustom

