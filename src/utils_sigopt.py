import threading
from sigopt import Connection
import subprocess

api_key = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'
client_id = 10179

def evaluate_model(assignment, gpu, name):
  assignment = dict(assignment)
  command = 'python main.py --train --transfer --beamsearch' + \
            ' --gpu=' + str(gpu) + \
            ' --name=' + name + ' ' + \
            ' '.join(['--' + k +'=' + str(v) for k,v in assignment.items()])
  print(command)
  output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
  i = output.stdout.find('bestCharErrorRate')
  charAccuracy = 1-float(output.stdout[(i+18):(i+32)])
  print('Suggestion', name, 'charAccuracy', charAccuracy)
  return charAccuracy # optimization metric is the char accuracy

class Master(threading.Thread):
  '''master thread which initializes an experimetn and starts the workers (passing in the expt id and gpu it should run on.
  doesnt seem necessary for this to be its own class nor does it seem necessary for it to inhereit the threading class.'''

  def __init__(self, exptDetail, name, gpus, exptId=None, resume=False, **kwargs):
    threading.Thread.__init__(self) # required: call constructor of threading class
    self.gpus = gpus
    self.conn = Connection(client_token=api_key) # start connection to sigopt
    if exptId==None: # start anew experiment
      expt = self.conn.experiments().create(**exptDetail)
      exptId = expt.id
    else: # use existing experiment id
      expt = self.conn.experiments(exptId)
      expt.update(**exptDetail)
      if not resume: # delete all observations and suggestions
        expt.observations().delete()
        expt.suggestions().delete()
    self.name = name
    self.exptId = exptId # save expt id in the instance
    print("View your experiment progress: https://sigopt.com/experiment/{}".format(self.exptId))

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.exptId).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    '''start the workers and evaluations'''
    tries = 3
    while (tries > 0 and self.remaining_observations > 0):
      workers = [Worker(exptId=self.exptId, name=self.name, gpu=gpu) for gpu in self.gpus]
      for worker in workers:
        worker.start() # each worker will communicated spearately with sigopt and will not communicate wiht one another
      for worker in workers:
        worker.join() # wait till all workers are done, which only happens when observation budget runs out
      self.conn.experiments(self.exptId).suggestions().delete(state='open')
      tries -= 1 # will reach this point only when observation budget runs out or all workers have failed

class Worker(threading.Thread):
  '''worker thread which operates independently of all other workers and communicates only with sigopt server for
  suggestions and sends observations'''

  def __init__(self, exptId, name, gpu):
    threading.Thread.__init__(self)
    self.conn = Connection(client_token=api_key)
    self.exptId = exptId
    self.name = name
    self.gpu = gpu

  @property
  def metadata(self):
    return dict(host=threading.current_thread().name)

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.exptId).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    '''get suggetsions, run evaluation, report observation'''
    while self.remaining_observations > 0:
      suggestion = self.conn.experiments(self.exptId).suggestions().create(metadata=self.metadata)
      try:
        value = evaluate_model(assignment=suggestion.assignments, gpu=self.gpu, name=self.name+'-'+str(suggestion.id))
        failed = False
      except Exception:
        value = None
        failed = True
      self.conn.experiments(self.exptId).observations().create(suggestion=suggestion.id,
                                                               value=value,
                                                               failed=failed,
                                                               metadata=self.metadata,
                                                               )
      print(self.metadata, 'failed='+str(failed))

def get_expt_ids():
  return conn.clients(client_id).plan().fetch().current_period.experiments

if __name__=='__main__':
  conn = Connection(client_token=api_key)
  experimentIds = conn.clients(client_id).plan().fetch().current_period.experiments
  expt = conn.experiments(experimentIds[0])
  expt.update(name='htr',
              parameters=parameters,
              observation_budget=len(parameters) * 30,
              parallel_bandwidth=bandwidth,
              )
  expt.observations().delete()
  expt.suggestions().delete()

