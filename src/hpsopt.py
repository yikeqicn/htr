import argparse
from utils_sigopt import Master
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='debug', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--exptId', default=None, type=int, help='existing experiment id?')
parser.add_argument('--gpus', default=[0], type=int, nargs='+')
parser.add_argument('--bandwidth', default=None, type=int)
args = parser.parse_args()
open('/root/hpsopt.bash','w+').write('nohup python '+' '.join(sys.argv)+' &\n') # write command to the log

api_key = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'

parameters = [dict(name='lrInit', type='double', bounds=dict(min=1e-3, max=5e-1)),
              dict(name='lrDrop1', type='int', bounds=dict(min=3, max=100)),
              dict(name='lrDrop2', type='int', bounds=dict(min=200, max=5000)),
              dict(name='wdec', type='double', bounds=dict(min=.1e-4, max=10e-4)),
              dict(name='optimizer', type='categorical', categorical_values=[dict(name='adam'), dict(name='rmsprop'), dict(name='momentum')]),
              dict(name='crop_r1', type='int', bounds=dict(min=0, max=6)),
              dict(name='crop_r2', type='int', bounds=dict(min=24, max=32)),
              dict(name='crop_c1', type='int', bounds=dict(min=6, max=14)),
              dict(name='crop_c2', type='int', bounds=dict(min=111, max=119)),
              ]

exptDetail = dict(name=args.name, parameters=parameters, observation_budget=len(parameters) * 30,
                  parallel_bandwidth=len(args.gpus) if args.bandwidth==None else args.bandwidth)

if __name__ == '__main__':
  master = Master(exptDetail=exptDetail, **vars(args))
  master.start()
  master.join()