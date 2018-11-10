import os
from os.path import join, basename, dirname
<<<<<<< Updated upstream
=======
import matplotlib
matplotlib.use('TkAgg')
>>>>>>> Stashed changes
import matplotlib.pyplot as plt

def log_image(experiment, batch, text, savetag, ckptpath, counter, epoch):
  imageFile = join(ckptpath, 'images', savetag+'-'+str(counter)+'.'+str(epoch)+'.jpg')
  os.makedirs(dirname(imageFile), exist_ok=True)
  plt.imshow(batch.imgs[counter].T, cmap='gray'); plt.axis('image'); plt.axis('tight'); plt.title(text);
  # plt.xticks([]); plt.yticks([]);
  plt.tight_layout(pad=0); plt.savefig(imageFile)
  experiment.log_image(imageFile)
