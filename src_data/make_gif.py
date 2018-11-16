# ---------------------------------
# MAKE GIF FROM IMAGES IN A DIRECTORY
# Example
# python make_gif.py <directory_of_images> <output_filename.gif>


import imageio
import numpy as np
import os
from glob import glob
from os.path import join
import sys
import random
import PIL.Image as Image
from multiprocessing import Pool

path = sys.argv[1]
if len(sys.argv)>2:
  outFile = sys.argv[2]
else:
  outFile = 'my_gif.gif'

print('==============> Making gif from files in '+path)
list_images = glob(join(path, '*'))
random.shuffle(list_images)
images = []
count = 1
# for filename in list_images:
def parallel(filename):
  img = Image.open(filename)
  width, height = img.size
  img = img.resize((105,21), Image.ANTIALIAS)
  img = np.array(img)
  # images.append(img)
  # print('Image '+str(count)+' of '+str(len(list_images)))
  # count+=1
  print('done with one')
  return img

pool = Pool(processes=4)
images = pool.map(parallel, list_images)

print('Combining images into gif')
imageio.mimsave(outFile, images, 'GIF', duration=.2)
print('gif saved in '+str(outFile))
