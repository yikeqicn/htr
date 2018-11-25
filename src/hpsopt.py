import argparse
from utils_sigopt import Master
from utils import maybe_download
import sys
import os
from shutil import rmtree

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='debug', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--exptId', default=None, type=int, help='existing experiment id?')
parser.add_argument('--gpus', default=[0], type=int, nargs='+')
parser.add_argument('--bandwidth', default=None, type=int)
args = parser.parse_args()
open('/root/hpsopt.bash','w').write('cd /root/repo/htr/src && python '+' '.join(sys.argv)+'\n') # write command to the log

# download dataset
dataUrl = 'https://www.dropbox.com/s/e6d1w5roodv8mv3/htr_assets.zip?dl=0'
maybe_download(dataUrl, 'htr_assets', '/root/datasets', filetype='zip', force=False)
# rmtree('/root/datasets/tmp')
# os.rename('/root/datasets/htr_assets', '/root/datasets/tmp')
# os.rename('/root/datasets/tmp/htr_assets', '/root/datasets/htr_assets')
# rmtree('/root/datasets/tmp')

api_key = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'

parameters = [dict(name='lrInit', type='double', default_value=.1,   bounds=dict(min=1e-3, max=5e-1)),
              dict(name='lrDrop1', type='int',   default_value=10,   bounds=dict(min=3, max=100)),
              dict(name='lrDrop2', type='int',   default_value=1e4,  bounds=dict(min=200, max=5000)),
              dict(name='wdec', type='double',   default_value=1e-4, bounds=dict(min=.1e-4, max=10e-4)),
              # dict(name='crop_r1', type='int',   default_value=3,    bounds=dict(min=0, max=6)),
              # dict(name='crop_r2', type='int',   default_value=28,   bounds=dict(min=24, max=32)),
              # dict(name='crop_c1', type='int',   default_value=10,   bounds=dict(min=6, max=14)),
              # dict(name='crop_c2', type='int',   default_value=115,  bounds=dict(min=111, max=119)),
              # dict(name='optimizer', type='categorical', categorical_values=[dict(name='adam'), dict(name='rmsprop'), dict(name='momentum')]),
              ]

exptDetail = dict(name=args.name, parameters=parameters, observation_budget=len(parameters) * 30,
                  parallel_bandwidth=len(args.gpus) if args.bandwidth==None else args.bandwidth)

if __name__ == '__main__':
  master = Master(exptDetail=exptDetail, **vars(args))
  master.start()
  master.join()