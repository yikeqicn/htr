import os
from os.path import join, basename, dirname
import matplotlib.pyplot as plt
import time

def log_image(experiment, batch, text, savetag, ckptpath, counter, epoch):
  imageFile = join(ckptpath, 'images', savetag+'-'+str(counter)+'.'+str(epoch)+'.jpg')
  os.makedirs(dirname(imageFile), exist_ok=True)
  plt.imshow(batch.imgs[counter].T, cmap='gray'); plt.axis('image'); plt.title(text.replace('$','\$'));
  # plt.axis('tight')
  plt.tight_layout(pad=0); time.sleep(.9)
  plt.savefig(imageFile)
  experiment.log_image(imageFile)
