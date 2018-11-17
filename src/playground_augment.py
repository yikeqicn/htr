import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis
from PIL import Image
import os
from os.path import join, basename, dirname
from glob import glob
from src.SamplePreprocessor import horizontal_stretch, keep_aspect_pad, target_aspect_pad
from numpy.random import choice, normal, rand
from utils_preprocess import *

imshow = lambda im: Image.fromarray(im).show()
home = os.environ['HOME']

if __name__ == '__main__':
  htrAssetsRoot = join(home, 'datasets', 'htr_assets')
  crowdRoot = join(htrAssetsRoot, 'crowdsource')
  processedRoot = join(crowdRoot, 'processed')
  patchBoxesRoot = join(htrAssetsRoot, 'cropped_patches', 'nw_boxes-3')
  patchHorizRoot = join(htrAssetsRoot, 'cropped_patches', 'nw_horizontal-2')
  testsetRoot = join(htrAssetsRoot, 'nw_im_crop_curated')

  imgSize = (128*4, 32*4)
  imgSize = (128, 32)
  baseFiles = glob(join(processedRoot, '*/*.jpg'))
  patchBoxesFiles = glob(join(patchBoxesRoot, '*.jpg'))
  patchHorizFiles = glob(join(patchHorizRoot, '*.jpg'))
  for file in np.array(baseFiles)[np.random.permutation(len(baseFiles))[:10]]:

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = horizontal_stretch(img, minFactor=.5, maxFactor=1.5)
    img = target_aspect_pad(img, targetRatio=imgSize[1] / imgSize[0])
    img = keep_aspect_pad(img, maxFactor=1.2)
    img = cv2.resize(img, imgSize, interpolation=cv2.INTER_CUBIC)
    if rand() < .90: img = merge_patch_box_random(img, centroid_std=.1)
    else: img = merge_patch_horiz_random(img, centroid_std=.05)
    imshow(img)

