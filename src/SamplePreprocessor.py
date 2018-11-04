import random
import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis
from numpy.random import randint


def horizontal_stretch(img, minFactor, maxFactor):
  '''randomly stretch image horizontally by amount uniformly between minFactor and maxFactor'''
  return cv2.resize(img, (int(img.shape[1] * np.random.uniform(minFactor, maxFactor)), img.shape[0]))


def target_aspect_pad(img, targetRatio=32 / 128):
  '''change aspect ratio of image to targetRatio by padding one of the dimensions. original image will be placed in
  random location within the expanded canvas'''
  nr, nc = img.shape
  currentRatio = nr / nc
  if currentRatio > targetRatio:
    dc = int(nr * (1 / targetRatio - 1 / currentRatio))
    dc2 = randint(max(dc, 1))
    dc1 = dc - dc2
    padding = ((0, 0), (dc1, dc2))
  else:
    dr = int(nc * (targetRatio - currentRatio))
    dr2 = randint(max(dr, 1))
    dr1 = dr - dr2
    padding = ((dr1, dr2), (0, 0))
  img = np.pad(img, padding, 'maximum')
  return img


def keep_aspect_pad(img, maxFactor):
  '''pad image by such that it expands by rand(maxFactor) while keeping its aspect ratio fixed. original image will be
  placed in random location within tthe expanded canvas. maxFactor must be greater than 1'''
  nr, nc = img.shape
  ratio = nr / nc
  dc = randint(max(int((maxFactor - 1) * nc), 1))
  dr = int(ratio * dc)
  dc2 = randint(max(dc, 1))
  dc1 = dc - dc2
  dr2 = randint(max(dr, 1))
  dr1 = dr - dr2
  padding = ((dr1, dr2), (dc1, dc2))
  img = np.pad(img, padding, 'maximum')
  return img


def preprocess(img, imgSize, args, dataAugmentation=False):
  "put img into target img of size imgSize, transpose for TF and normalize gray-values"

  # there are damaged files in IAM dataset - just use black image instead
  if img is None or np.min(img.shape) <= 1:
    img = np.zeros([imgSize[1], imgSize[0]])

  # increase dataset size by applying random stretches to the images
  if not args.custom:

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

  if args.custom:

    if dataAugmentation:
    # ADDED by ronny
      img = horizontal_stretch(img, minFactor=.7, maxFactor=1.5)
      img = target_aspect_pad(img, targetRatio=imgSize[1]/imgSize[0])
      img = keep_aspect_pad(img, maxFactor=1.5)

    target = cv2.resize(img, imgSize, interpolation=cv2.INTER_CUBIC)

  # transpose for TF
  img = cv2.transpose(target)

  # normalize
  (m, s) = cv2.meanStdDev(img)
  m = m[0][0]
  s = s[0][0]
  img = img - m
  img = img / s if s > 0 else img

  return img

