import threading
from sigopt import Connection
import subprocess

# You can find your API token at https://sigopt.com/docs/overview/authentication
SIGOPT_API_KEY = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'

parameters = [dict(name='lrInit', type='double', bounds=dict(min=1e-5, max=5e-1)),
              dict(name='wdec', type='double', bounds=dict(min=.1e-4, max=9e-4)),
              dict(name='optimizer', type='categorical', categorical_values=[dict(name='adam'), dict(name='rmsprop'), dict(name='momentum')]),
              dict(name='crop_r1', type='int', bounds=dict(min=0, max=6)),
              dict(name='crop_r2', type='int', bounds=dict(min=24, max=32)),
              dict(name='crop_c1', type='int', bounds=dict(min=6, max=14)),
              dict(name='crop_c2', type='int', bounds=dict(min=111, max=119)),
              ]

def evaluate_model(assignment, gpu):
  assignment = dict(assignment)
  command = 'python main.py --train --beamsearch --gpu='+str(gpu)+' '+' '.join(['--' + k +'=' + str(v) for k,v in assignment.items()])
  print(command)
  output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
  id = output.stdout.find('bestCharErrorRate')
  return float(output.stdout[(id+18):(id+32)])

n_gpus = 4
class Master(threading.Thread):
  """
  Shows what a master machine does when running SigOpt in a distributed setting.
  """

  def __init__(self):
    """
    Initialize the master thread, creating the SigOpt API connection and the experiment.
    We use the observation_budget field on the experiment to keep track of approximately
    how many total Observations we want to report. We recommend using a number between 10-20x
    the number of parameters in an experiment.
    """
    threading.Thread.__init__(self)
    self.conn = Connection(client_token=SIGOPT_API_KEY)
    experiment = self.conn.experiments().create(name='Parallel Experiment',
                                                parameters=parameters,
                                                observation_budget=len(parameters) * 20,
                                                )
    print("View your experiment progress: https://sigopt.com/experiment/{}".format(experiment.id))
    self.experiment_id = experiment.id

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    """
    Attempt to run NUM_WORKERS worker machines. If any machines fail, retry up to
    three times, deleting openSuggestions before proceeding.
    """
    tries = 3
    while (tries > 0 and self.remaining_observations > 0):
      workers = [Worker(self.experiment_id, gpu) for gpu in range(n_gpus)]
      for worker in workers:
        worker.start()
      for worker in workers:
        worker.join()
      self.conn.experiments(self.experiment_id).suggestions().delete(state='open')
      tries -= 1

class Worker(threading.Thread):

  def __init__(self, experiment_id, gpu):
    """
    Initialize a worker thread, creating the SigOpt API connection and storing the previously
    created experiment's id
    """
    threading.Thread.__init__(self)
    self.conn = Connection(client_token=SIGOPT_API_KEY)
    self.experiment_id = experiment_id
    self.gpu = gpu

  @property
  def metadata(self):
    """
    Use metadata to keep track of the host that each Suggestion and Observation is created on.
    Learn more: https://sigopt.com/docs/overview/metadata
    """
    return dict(host=threading.current_thread().name)

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    """
    SigOpt acts as the scheduler for the Suggestions, so all you need to do is run the
    optimization loop until there are no remaining Observations to be reported.
    We handle exceptions by reporting failed Observations. Learn more about handling
    failure cases: https://sigopt.com/docs/overview/metric_failure
    """
    while self.remaining_observations > 0:
      suggestion = self.conn.experiments(self.experiment_id).suggestions().create(metadata=self.metadata)
      try:
        value = evaluate_model(suggestion.assignments, self.gpu)
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
  master = Master()
  master.start()
  master.join()