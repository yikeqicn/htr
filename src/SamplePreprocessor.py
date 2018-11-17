import random
import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis
from numpy.random import randint
import os
from src.utils_preprocess import *

def preprocess(img, imgSize, args, dataAugmentation=False):
  "put img into target img of size imgSize, transpose for TF and normalize gray-values"

  # there are damaged files in IAM dataset - just use black image instead
  if img is None or np.min(img.shape) <= 1:
    img = np.zeros([imgSize[1], imgSize[0]])

  # increase dataset size by applying random stretches to the images
  if args.noncustom:

    if dataAugmentation:
      stretch = (random.random() - 0.5)  # -0.5 .. +0.5
      wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
      img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

  else:

    if dataAugmentation:
      img = horizontal_stretch(img, minFactor=.5, maxFactor=1.5)
      img = target_aspect_pad(img, targetRatio=imgSize[1] / imgSize[0])
      img = keep_aspect_pad(img, maxFactor=1.2)
      img = cv2.resize(img, tuple(imgSize), interpolation=cv2.INTER_CUBIC)
      if rand() < .70: img = merge_patch_box_random(img, centroid_std=.025)
      else: img = merge_patch_horiz_random(img, centroid_std=.05)

  target = cv2.resize(img, tuple(imgSize), interpolation=cv2.INTER_CUBIC)

  # transpose for TF
  img = cv2.transpose(target)

  # normalize
  (m, s) = cv2.meanStdDev(img)
  m = m[0][0]
  s = s[0][0]
  img = img - m
  img = img / s if s > 0 else img

  return img
