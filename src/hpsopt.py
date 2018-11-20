import sigopt
import subprocess

class Objective(object):
  """The Objective function to maximize.

  Replace with any function or time consuming and expensive process.
  """

  def __init__(self):
    self.parameters = [dict(name='lrInit', type='double', bounds=dict(min=1e-5, max=5e-1)),
                       dict(name='wdec', type='double', bounds=dict(min=.1e-4, max=9e-4)),
                       dict(name='optimizer', type='categorical', categorical_values=[dict(name='adam'), dict(name='rmsprop'), dict(name='momentum')]),
                       dict(name='crop_r1', type='int', bounds=dict(min=0, max=6)),
                       dict(name='crop_r2', type='int', bounds=dict(min=24, max=32)),
                       dict(name='crop_c1', type='int', bounds=dict(min=6, max=14)),
                       dict(name='crop_c2', type='int', bounds=dict(min=111, max=119)),
                       ]

  @staticmethod
  def evaluate(params):
    command = ['python main.py']+['--'+k+'='+str(v) for k,v in params.items()]
    command = 'python main.py --train --beamsearch --gpu=0 '+' '.join(['--'+k+'='+str(v) for k,v in params.items()])
    print('===================== START EVALUATION ======================')
    print(command)
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')






# create objective instance
obj = Objective()

# create sigopt connection and experiment
conn = sigopt.Connection(client_token="FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL")
experiment = conn.experiments().create(name='htr-first', parameters=obj.parameters, observation_budget=10*len(obj.parameters))
print("Sigopt experiment at https://sigopt.com/experiment/{0}".format(experiment.id))

# loop through cycle of observation, suggestion, and evaluation
for _ in range(experiment.observation_budget):

  # get suggestion
  suggestion = conn.experiments(experiment.id).suggestions().create()

  # evaluate output of that suggestion
  value = obj.evaluate(suggestion.assignments)

  # return evaluated value to sigopt as an observation
  conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=value)

# after optimization loop, get best ones
assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
print(assignments)

# This is a SigOpt-tuned model
with tf.Session() as sess:
  classifier.create_model(assignments, sess)




