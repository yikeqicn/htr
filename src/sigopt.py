

class Objective(object):
  """The Objective function to maximize.

  Replace with any function or time consuming and expensive process.
  """

  def __init__(self):
    self.parameters = [
      {'name': 'x1',
       'type': 'double',
       'bounds': {'min': -70.0, 'max': 70.0},
       },
      {'name': 'x2',
       'type': 'double',
       'bounds': {'min': -70.0, 'max': 70.0},
       },
    ]

  @staticmethod
  def evaluate(params):
    subprocess.run('python main.py',)

obj = Objective()
conn = sigopt.Connection(client_token="FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL")

experiment = conn.experiments().create(
  name='htr-first',
  parameters=obj.parameters,
  observation_budget=40,
)

print("Sigopt experiment at https://sigopt.com/experiment/{0}".format(experiment.id))