

class Objective(object):
  """The Objective function to maximize.

  Replace with any function or time consuming and expensive process.
  """

  def __init__(self):
    self.parameters = [
      {'name': 'lrInit', 'type': 'double', 'bounds': {'min': 1e-5, 'max': 5e-1},
       },
      {'name': 'optimizer',
       'type': 'categorical',
       'bounds': {'min': -70.0, 'max': 70.0},
       },
    ]

  @staticmethod
  def evaluate(params):
    command = ['python main.py']+['--'+k+'='+str(v) for k,v in params.items()]
    subprocess.run(command)
)

obj = Objective()
conn = sigopt.Connection(client_token="FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL")

experiment = conn.experiments().create(
  name='htr-first',
  parameters=obj.parameters,
  observation_budget=10*len(self.parameters),
)

print("Sigopt experiment at https://sigopt.com/experiment/{0}".format(experiment.id))