import random
import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis
from numpy.random import randint
import os


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

def remove_lines(img):
  '''remove straight lines (of assumed infinite length) from image'''
  img_copy = img.copy()
  gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

  # gahther dimensions
  largerDim = np.max(gray.shape)
  origShape = gray.shape
  sqrShape = [largerDim, largerDim]

  gray = cv2.resize(gray, (largerDim, largerDim)) # resize to square image to ensure the votes count equally
  # gray = cv2.GaussianBlur(gray,(3,3),0)

  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  lines = cv2.HoughLines(edges, 2, 3 * np.pi / 180, threshold=300)
  # plot(*np.squeeze(lines).T, '.'); show() # plot hough space

  # apply lines to image
  for rho, theta in np.squeeze(lines):

    # calculate line enedpoints
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # resize back down the endpoints to original image dimensions
    x1 = int( x1 * origShape[1]/sqrShape[1] )
    x2 = int( x2 * origShape[1]/sqrShape[1] )
    y1 = int( y1 * origShape[0]/sqrShape[0] )
    y2 = int( y2 * origShape[0]/sqrShape[0] )

    # apply the line to the image
    cv2.line(img_copy, (x1, y1), (x2, y2), (255, 255, 255), thickness=4)
  return img_copy

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

    if True: # dataAugmentation
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

