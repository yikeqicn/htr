from __future__ import print_function
import os
from tqdm import tqdm
import random
import numpy as np
from mnist_sequence_api import MNIST_Sequence_API
import cv2
from random import randint

api_object = MNIST_Sequence_API()

num_samples = 500000
outputdir = "/data/home/jdegange/vision/digitsdataset2/"


def noisy(noise_typ,image):
    if noise_typ == 1: #"gauss"
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == 2: #"s&p"
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == 3: #"poisson"
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ ==4: #"speckle"
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy


print("write all images")
count = 0
for i in tqdm(range(num_samples)):
  count = count +1
  seq_len = np.random.random_integers(8)
  sqnc = np.random.randint(0, 10, seq_len)
  seq = api_object.save_image(api_object.generate_mnist_sequence(sqnc, (0,10),28 * seq_len),sqnc)

os.chdir(outputdir)
print("resize and add noise")
for filename in tqdm(os.listdir(outputdir)):
  #print(filename)
  im = cv2.imread(filename,cv2.IMREAD_COLOR) 
  large = cv2.resize(im, (0,0), fx=3, fy=3) #300% larger
  randcoord = randint(-35, 35)
  randthick = randint(1, 5)
  cv2.line(large, (0, int(41+randcoord)), (int(large.shape[1]), int(41+randcoord+randthick)), (0,0,0), randthick)
  cv2.cvtColor(large, cv2.COLOR_BGR2GRAY) #grayscale
  cv2.imwrite(filename,noisy(np.random.random_integers(4),large))  #random noise