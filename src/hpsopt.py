import threading
from sigopt import Connection
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_id", default=None, type=int, help="existing experiment id?")
parser.add_argument("--gpus", default=[0,0,1,1,2,2,3,3], type=int, nargs='+')
parser.add_argument("--bandwidth", default=None, type=int)
args = parser.parse_args()

SIGOPT_API_KEY = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'

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

def evaluate_model(assignment, gpu, name):
  assignment = dict(assignment)
  command = 'python main.py --train --beamsearch' + \
            ' --gpu=' + str(gpu) + \
            ' --name=' + name + ' ' + \
            ' '.join(['--' + k +'=' + str(v) for k,v in assignment.items()])
  print(command)
  output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
  i = output.stdout.find('bestCharErrorRate')
  return 1-float(output.stdout[(i+18):(i+32)]) # char accuracy


class Master(threading.Thread):

  def __init__(self, experiment_id=None, bandwidth=None):
    threading.Thread.__init__(self)
    bandwidth = len(args.gpus) if bandwidth==None
    if experiment_id==None:
      self.conn = Connection(client_token=SIGOPT_API_KEY)
      experiment = self.conn.experiments().create(name='htr',
                                                  parameters=parameters,
                                                  observation_budget=len(parameters) * 20,
                                                  parallel_bandwidth=bandwidth,
                                                  )
      experiment_id = experiment.id

    self.experiment_id = experiment_id
    print("View your experiment progress: https://sigopt.com/experiment/{}".format(self.experiment_id))

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    tries = 3
    while (tries > 0 and self.remaining_observations > 0):
      workers = [Worker(self.experiment_id, gpu) for gpu in args.gpus]
      for worker in workers:
        worker.start()
      for worker in workers:
        worker.join()
      self.conn.experiments(self.experiment_id).suggestions().delete(state='open')
      tries -= 1

class Worker(threading.Thread):

  def __init__(self, experiment_id, gpu):
    threading.Thread.__init__(self)
    self.conn = Connection(client_token=SIGOPT_API_KEY)
    self.experiment_id = experiment_id
    self.gpu = gpu

  @property
  def metadata(self):
    return dict(host=threading.current_thread().name)

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    while self.remaining_observations > 0:
      suggestion = self.conn.experiments(self.experiment_id).suggestions().create(metadata=self.metadata)
      try:
        value = evaluate_model(suggestion.assignments, self.gpu, suggestion.id)
        failed = False
      except Exception:
        value = None
        failed = True
      self.conn.experiments(self.experiment_id).observations().create(suggestion=suggestion.id,
                                                                      value=value,
                                                                      failed=failed,
                                                                      metadata=self.metadata,
                                                                      )

if __name__ == '__main__':
  master = Master(experiment_id=args.experiment_id, bandwidth=args.bandwidth)
  master.start()
  master.join()